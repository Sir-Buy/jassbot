// ==UserScript==
// @name         Swisslos Jass Collector
// @namespace    jass-collector
// @version      8.0.0
// @description  Jass Differenzler assistant — WATCH / ASSIST / BOT modes, PNL tracking
// @match        https://www.swisslos.ch/de/jass/differenzler/spielen.html
// @match        https://www.swisslos.ch/fr/jass/differenzler/jouer.html
// @match        https://www.swisslos.ch/it/jass/differenzler/giocare.html
// @match        https://www.swisslos.ch/en/jass/differenzler/play.html
// @grant        none
// @run-at       document-start
// ==/UserScript==

(function () {
  'use strict';

  // ═══════════════════════════════════════════════════════════════════════
  //  CONFIG
  // ═══════════════════════════════════════════════════════════════════════
  const AUTO_JOIN_DELAY = 4000;
  const CYCLE_DELAY = 5000;
  const MATCH_END_TIMEOUT = 20000;
  const STORAGE_WARN_BYTES = 4 * 1024 * 1024;
  const STORAGE_EXPORT_BYTES = 4.2 * 1024 * 1024;
  const KEEP_AFTER_EXPORT = 50;
  const BOT_SERVER_URL = 'http://localhost:5000';
  const BOT_PLAY_DELAY_MIN = 1000;
  const BOT_PLAY_DELAY_MAX = 3000;
  const SERVER_CHECK_INTERVAL = 10000;

  // ═══════════════════════════════════════════════════════════════════════
  //  TAB NUMBERING
  // ═══════════════════════════════════════════════════════════════════════
  let tabNumber = parseInt(sessionStorage.getItem('jc_tab_number')) || 0;
  if (!tabNumber) {
    let activeTabs = [];
    try { activeTabs = JSON.parse(localStorage.getItem('jc_active_tabs') || '[]'); } catch {}
    activeTabs = activeTabs.filter(t => Date.now() - t.ts < 30000);
    tabNumber = activeTabs.length + 1;
    sessionStorage.setItem('jc_tab_number', tabNumber);
  }
  function updateTabHeartbeat() {
    let activeTabs = [];
    try { activeTabs = JSON.parse(localStorage.getItem('jc_active_tabs') || '[]'); } catch {}
    activeTabs = activeTabs.filter(t => Date.now() - t.ts < 30000 && t.id !== tabNumber);
    activeTabs.push({ id: tabNumber, ts: Date.now() });
    try { localStorage.setItem('jc_active_tabs', JSON.stringify(activeTabs)); } catch {}
  }
  updateTabHeartbeat();
  setInterval(updateTabHeartbeat, 10000);

  function updateTitle() {
    document.title = 'Tab ' + tabNumber + ' \u2014 Jass Collector';
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  CARD SYSTEM  (card_id = suit * 9 + value)
  // ═══════════════════════════════════════════════════════════════════════
  const SUIT_ID   = { H: 0, D: 1, S: 2, C: 3 };
  const SUIT_NAME = { H: 'Herz', D: 'Ecke', S: 'Schaufel', C: 'Kreuz' };
  const SUIT_SYM  = { H: '\u2665', D: '\u2666', S: '\u2660', C: '\u2663' };
  const SUIT_CLR  = { H: '#e53935', D: '#e53935', S: '#64b5f6', C: '#b0b0b0' };
  const VAL_ID    = { '6': 0, '7': 1, '8': 2, '9': 3, '10': 4, J: 5, Q: 6, K: 7, A: 8 };
  const BASE_PTS  = { '6': 0, '7': 0, '8': 0, '9': 0, '10': 10, J: 2, Q: 3, K: 4, A: 11 };
  const TRUMP_PTS = { '6': 0, '7': 0, '8': 0, '9': 14, '10': 10, J: 20, Q: 3, K: 4, A: 11 };

  function parseCard(code) {
    if (!code || typeof code !== 'string') return null;
    const m = code.trim().match(/^([DHSC])(\d+|[JQKA])$/);
    if (!m) return null;
    const s = SUIT_ID[m[1]], v = VAL_ID[m[2]];
    if (s === undefined || v === undefined) return null;
    return { raw: code, suit: s, value: v, id: s * 9 + v, suitChar: m[1], valStr: m[2] };
  }

  function cardPoints(code, trumpSuit) {
    if (!code) return 0;
    const s = code.charAt(0), v = code.substring(1);
    return (s === trumpSuit ? TRUMP_PTS[v] : BASE_PTS[v]) || 0;
  }

  function cardDisplay(code) {
    if (!code) return '?';
    const s = code.charAt(0), v = code.substring(1);
    return (SUIT_SYM[s] || s) + v;
  }

  function cardColorStyle(code) {
    if (!code) return 'color:#ccc';
    const s = code.charAt(0);
    return 'color:' + (SUIT_CLR[s] || '#ccc');
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  STORAGE
  // ═══════════════════════════════════════════════════════════════════════
  const STOR = {
    matches: 'jc_matches',
    tables: 'jc_tables',
    stats: 'jc_stats',
  };

  function defaultStats() {
    return {
      started: Date.now(), rounds: 0, matches: 0, fullRounds: 0,
      totalDev: 0, devHist: [], trumpDist: [0, 0, 0, 0], pointsHist: [],
      perfectRounds: 0, roomDist: {},
    };
  }

  function loadMatches() {
    try { return JSON.parse(localStorage.getItem(STOR.matches)) || []; } catch { return []; }
  }
  function saveMatches(m) {
    try { localStorage.setItem(STOR.matches, JSON.stringify(m)); } catch (e) { log('Storage error: ' + e.message); }
  }
  function loadStats() {
    try { return JSON.parse(localStorage.getItem(STOR.stats)) || defaultStats(); } catch { return defaultStats(); }
  }
  function saveStats(s) {
    try { localStorage.setItem(STOR.stats, JSON.stringify(s)); } catch {}
  }

  function getStorageUsage() {
    let total = 0;
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        total += (k.length + localStorage.getItem(k).length) * 2;
      }
    } catch {}
    return total;
  }

  function checkStorageCapacity() {
    const used = getStorageUsage();
    if (used >= STORAGE_EXPORT_BYTES) {
      log('Storage critical (' + (used / 1048576).toFixed(1) + 'MB). Auto-exporting...');
      exportJSON();
      const matches = loadMatches();
      if (matches.length > KEEP_AFTER_EXPORT) {
        saveMatches(matches.slice(-KEEP_AFTER_EXPORT));
        log('Trimmed to last ' + KEEP_AFTER_EXPORT + ' matches after auto-export.');
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  TABLE DISCOVERY
  // ═══════════════════════════════════════════════════════════════════════
  let knownTables = {};
  let roomInfo = {};

  function onTableUpdate(data) {
    if (!data || !data.t_id) return;
    knownTables[String(data.t_id)] = {
      mt_id: data.mt_id, state: data.state || '?',
      players: data.p ? data.p.filter(p => p && p.n).map(p => p.n) : [],
      ts: Date.now(),
    };
    try { localStorage.setItem(STOR.tables, JSON.stringify(knownTables)); } catch {}
  }

  function onRoomList(data) {
    if (!data || !data.id) return;
    data.id.forEach((id, i) => {
      roomInfo[id] = {
        title: data.title[i], stake: data.stake[i],
        pot: data.matchPot ? data.matchPot[i] : null,
        rounds: data.rounds ? data.rounds[i] : 4,
      };
    });
    log('Rooms: ' + data.title.join(', '));
  }

  function pickNextTable() {
    const now = Date.now();
    try {
      const shared = JSON.parse(localStorage.getItem(STOR.tables)) || {};
      for (const [t, info] of Object.entries(shared))
        if (!knownTables[t] || shared[t].ts > knownTables[t].ts) knownTables[t] = info;
    } catch {}
    const cands = [];
    for (const [tid, info] of Object.entries(knownTables)) {
      if (now - info.ts > 120000) continue;
      if (info.state === 'play') cands.push({ t_id: tid, mt_id: info.mt_id });
    }
    if (!cands.length) {
      for (const [tid, info] of Object.entries(knownTables)) {
        if (now - info.ts > 120000) continue;
        if (info.state === 'wait' && info.players && info.players.length >= 2)
          cands.push({ t_id: tid, mt_id: info.mt_id });
      }
    }
    return cands.length ? cands[Math.floor(Math.random() * cands.length)] : null;
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  PLAYER TRACKING
  // ═══════════════════════════════════════════════════════════════════════
  let allPlayers = {};

  function updatePlayerStats() {
    allPlayers = {};
    const matches = loadMatches();
    matches.forEach(match => {
      if (!match.players || !match.rounds) return;
      match.rounds.forEach(round => {
        if (!round.results || !round.results.results) return;
        round.results.results.forEach(res => {
          const player = match.players.find(p => p.pos === res.pos);
          const key = player ? (player.uid || player.name) : 'seat' + res.pos;
          const name = player ? player.name : 'Seat ' + res.pos;
          if (!allPlayers[key]) {
            allPlayers[key] = {
              name: name, rounds: 0, totalDev: 0, bestDev: 999,
              perfectCount: 0, totalDecl: 0, declCount: 0, deviations: [],
            };
          }
          const dev = Math.abs((res.callP || 0) - (res.points || 0));
          allPlayers[key].rounds++;
          allPlayers[key].totalDev += dev;
          if (dev < allPlayers[key].bestDev) allPlayers[key].bestDev = dev;
          if (dev === 0) allPlayers[key].perfectCount++;
          if (res.callP !== undefined) {
            allPlayers[key].totalDecl += res.callP;
            allPlayers[key].declCount++;
          }
          allPlayers[key].deviations.push(dev);
          if (allPlayers[key].deviations.length > 20) allPlayers[key].deviations.shift();
        });
      });
    });
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  GAME / MATCH STATE
  // ═══════════════════════════════════════════════════════════════════════
  let currentMatch = null;
  let currentRound = null;
  let currentTrick = null;
  let trickNumber = 0;
  let cardsInTrick = 0;
  let isInGame = false;
  let matchRound = 0;
  let maxRounds = 4;
  let trumpSuit = null;
  let matchEndTimer = null;
  let autoJoinTimer = null;
  let autoJoinEnabled = true;
  let isPaused = false;
  let lastCompletedTrick = null;
  let roundScores = { 0: 0, 1: 0, 2: 0, 3: 0 };

  // ═══════════════════════════════════════════════════════════════════════
  //  BOT PLAY STATE
  // ═══════════════════════════════════════════════════════════════════════
  let botPlayEnabled = localStorage.getItem('jc_bot_mode') === 'true';
  let assistantMode = localStorage.getItem('jc_assistant_mode') === 'true';
  let serverConnected = false;
  let seqNr = 0;
  let myHand = [];
  let myPos = -1;
  let lastBotThinkTime = null;
  let lastDeclThinkTime = null;
  let botPlayTimer = null;
  let lastRecommendation = null;  // {card_code, card_name, gap, time_ms}
  let sessionPNL = null;  // fetched from server

  function botRandomDelay() {
    return BOT_PLAY_DELAY_MIN + Math.random() * (BOT_PLAY_DELAY_MAX - BOT_PLAY_DELAY_MIN);
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  BOT SERVER COMMUNICATION
  // ═══════════════════════════════════════════════════════════════════════
  function botFetch(path, body) {
    return fetch(BOT_SERVER_URL + path, {
      method: body ? 'POST' : 'GET',
      headers: body ? { 'Content-Type': 'application/json' } : {},
      body: body ? JSON.stringify(body) : undefined,
    }).then(function (r) { return r.json(); });
  }

  function checkServerStatus() {
    botFetch('/status').then(function (resp) {
      if (!serverConnected) {
        serverConnected = true;
        log('[BOT] Server connected (' + (resp.engine || 'unknown') + ')');
        updateDashboard();
      }
      // Update PNL from server response
      if (resp.session) {
        sessionPNL = resp.session;
        updateDashboard();
      }
    }).catch(function () {
      if (serverConnected) {
        serverConnected = false;
        log('[BOT] Server disconnected');
        if (botPlayEnabled) {
          botPlayEnabled = false;
          log('[BOT] Bot Play disabled (server down)');
        }
        if (assistantMode) {
          assistantMode = false;
          log('[ASSIST] Assistant disabled (server down)');
        }
        updateDashboard();
      }
    });
  }

  function botNewRound(hand, trump) {
    return botFetch('/new-round', { hand: hand, trump: trump, my_pos: myPos });
  }

  function botPlay(hand, trickCards, leader, legalMoves, oppPoints) {
    return botFetch('/play', {
      hand: hand,
      trick_cards: trickCards,
      leader: leader,
      legal_moves: legalMoves,
      opp_points: oppPoints,
    });
  }

  function botNotifyTrick(trick, winner, points) {
    return botFetch('/notify-trick', { trick: trick, winner: winner, points: points });
  }

  function botEndRound(rank, opponents, matchId) {
    return botFetch('/end-round', {
      rank: rank || 0,
      opponents: opponents || [],
      match_id: matchId || '',
    });
  }

  function botSaveMatch(matchData) {
    return botFetch('/save', matchData);
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  BOT PLAY ACTIONS
  // ═══════════════════════════════════════════════════════════════════════
  function botHandleDeclaration(hand, trump) {
    if ((!botPlayEnabled && !assistantMode) || !serverConnected) return;
    log('[BOT] Requesting declaration...');
    botNewRound(hand, trump).then(function (resp) {
      var declaration = resp.declaration !== undefined ? resp.declaration : 0;
      lastDeclThinkTime = resp.time_ms || null;

      if (assistantMode && !botPlayEnabled) {
        // Show recommendation but don't auto-submit
        log('[ASSIST] Declare: ' + declaration + (lastDeclThinkTime ? ' (' + lastDeclThinkTime + 'ms)' : ''));
        // Show overlay with declaration recommendation
        var old = document.getElementById('jc-assist-overlay');
        if (old) old.remove();
        var overlay = document.createElement('div');
        overlay.id = 'jc-assist-overlay';
        overlay.style.cssText = 'position:fixed;top:12px;left:50%;transform:translateX(-50%);' +
          'background:linear-gradient(135deg,#0d1b2a,#1b2a3a);border:2px solid #4CAF50;' +
          'padding:16px 32px;border-radius:12px;z-index:999999;font-family:monospace;' +
          'box-shadow:0 4px 20px rgba(0,0,0,0.5);text-align:center;';
        overlay.innerHTML = '<div style="color:#4CAF50;font-size:24px;font-weight:bold;">' +
          'Declare: ' + declaration + '</div>' +
          '<div style="color:#aaa;font-size:12px;margin-top:4px;">' +
          (lastDeclThinkTime ? lastDeclThinkTime + 'ms' : '') + '</div>';
        document.body.appendChild(overlay);
        setTimeout(function () { if (overlay.parentNode) overlay.remove(); }, 15000);
      } else {
        log('[BOT] Declaration: ' + declaration + (lastDeclThinkTime ? ' (' + lastDeclThinkTime + 'ms)' : ''));
        sendBotMessage(1, { points: String(declaration) });
      }
      updateDashboard();
    }).catch(function (err) {
      log('[BOT] Declaration error: ' + err.message);
    });
  }

  function botHandlePlayCard(pCards) {
    if ((!botPlayEnabled && !assistantMode) || !serverConnected) return;

    var trickCards = [];
    if (currentTrick && currentTrick.cards) {
      trickCards = currentTrick.cards.map(function (c) {
        return { player: c.player, card: c.card };
      });
    }

    var leader = currentTrick ? currentTrick.leader : 0;
    var oppPoints = [];
    for (var s = 0; s < 4; s++) {
      if (s !== myPos) oppPoints.push(roundScores[s] || 0);
    }

    log('[BOT] Requesting play... (hand: ' + myHand.length + ', legal: ' + pCards.length + ')');
    botPlay(myHand, trickCards, leader, pCards, oppPoints).then(function (resp) {
      var cardCode = resp.card_code || pCards[0];
      lastBotThinkTime = resp.time_ms || null;
      lastRecommendation = {
        card_code: cardCode,
        card_name: resp.card_name || cardCode,
        gap: resp.gap,
        time_ms: resp.time_ms,
        hand: resp.hand || myHand,
        legal: resp.legal || pCards,
        target: resp.target,
        my_points: resp.my_points,
      };

      // ── ASSISTANT MODE: show recommendation, don't auto-play ──
      if (assistantMode && !botPlayEnabled) {
        log('[ASSIST] Play: ' + cardDisplay(cardCode) + (lastBotThinkTime ? ' (' + lastBotThinkTime + 'ms)' : ''));
        showAssistantOverlay(cardCode, pCards, resp);
        updateDashboard();
        return;
      }

      // ── BOT PLAY MODE: auto-play ──
      log('[BOT] Play: ' + cardDisplay(cardCode) + (lastBotThinkTime ? ' (' + lastBotThinkTime + 'ms)' : ''));

      // Remove card from hand
      var idx = myHand.indexOf(cardCode);
      if (idx >= 0) myHand.splice(idx, 1);

      // Send immediately against bots, delay against humans
      var isBotGame = currentMatch && currentMatch.players && currentMatch.players.some(function (p) {
        return p && p.uid && p.uid < 0;
      });
      if (isBotGame) {
        sendBotMessage(6, { card: cardCode });
        updateDashboard();
      } else {
        var delay = botRandomDelay();
        log('[BOT] Delay: ' + Math.round(delay) + 'ms');
        clearTimeout(botPlayTimer);
        botPlayTimer = setTimeout(function () {
          sendBotMessage(6, { card: cardCode });
          updateDashboard();
        }, delay);
      }
    }).catch(function (err) {
      log('[BOT] Play error: ' + err.message);
    });
  }

  // ── Assistant overlay: show recommended card prominently ──
  function showAssistantOverlay(recommended, legalCards, resp) {
    // Remove old overlay if exists
    var old = document.getElementById('jc-assist-overlay');
    if (old) old.remove();

    var overlay = document.createElement('div');
    overlay.id = 'jc-assist-overlay';
    overlay.style.cssText = 'position:fixed;top:12px;left:50%;transform:translateX(-50%);' +
      'background:linear-gradient(135deg,#0d1b2a,#1b2a3a);border:2px solid #6dd5fa;' +
      'padding:12px 24px;border-radius:12px;z-index:999999;font-family:monospace;' +
      'box-shadow:0 4px 20px rgba(0,0,0,0.5);display:flex;align-items:center;gap:16px;';

    // Build hand display with recommendation highlighted
    var handHtml = '<div style="display:flex;gap:4px;flex-wrap:wrap;">';
    var cards = resp.hand || legalCards;
    var isLegal = {};
    legalCards.forEach(function(c) { isLegal[c] = true; });

    cards.forEach(function (c) {
      var isRec = c === recommended;
      var isPlayable = isLegal[c];
      var bg = isRec ? '#4CAF50' : (isPlayable ? '#1a3a5c' : '#111');
      var border = isRec ? '2px solid #fff' : (isPlayable ? '1px solid #6dd5fa55' : '1px solid #333');
      var opacity = isPlayable ? '1' : '0.4';
      var sz = isRec ? 'font-size:18px;font-weight:bold;' : 'font-size:14px;';
      handHtml += '<span style="' + cardColorStyle(c) + ';background:' + bg +
        ';border:' + border + ';border-radius:6px;padding:4px 6px;opacity:' + opacity +
        ';' + sz + '">' + cardDisplay(c) + (isRec ? ' \u2190' : '') + '</span>';
    });
    handHtml += '</div>';

    var gapText = resp.gap !== undefined ? (resp.gap > 0 ? 'Need ' + resp.gap + ' more' : (resp.gap === 0 ? 'On target!' : Math.abs(resp.gap) + ' over')) : '';
    var infoHtml = '<div style="text-align:right;font-size:11px;color:#aaa;line-height:1.6;">' +
      'Target: ' + (resp.target || '?') + ' | Scored: ' + (resp.my_points || 0) +
      (gapText ? '<br>' + gapText : '') +
      '<br><span style="color:#6dd5fa;">' + (resp.time_ms || '?') + 'ms</span>' +
      '</div>';

    overlay.innerHTML = handHtml + infoHtml;
    document.body.appendChild(overlay);

    // Auto-remove after 15 seconds
    setTimeout(function () {
      if (overlay.parentNode) overlay.remove();
    }, 15000);
  }

  function botHandleTrickDone(trick, winner, points) {
    if ((!botPlayEnabled && !assistantMode) || !serverConnected) return;
    // Remove assistant overlay on trick completion
    var old = document.getElementById('jc-assist-overlay');
    if (old) old.remove();
    var trickCards = trick.cards.map(function (c) {
      return { player: c.player, card: c.card };
    });
    botNotifyTrick(trickCards, winner, points).catch(function (err) {
      log('[BOT] Trick notify error: ' + err.message);
    });
  }

  function botHandleRoundEnd() {
    if (!serverConnected) return;
    // Extract rank and opponent names from current round results
    var rank = 0;
    var opponents = [];
    var matchId = currentMatch ? currentMatch.match_id : '';
    if (currentRound && currentRound.results) {
      var resList = currentRound.results.results || currentRound.results;
      if (Array.isArray(resList)) {
        resList.forEach(function (r) {
          if (r.pos === myPos) rank = r.rank || 0;
        });
      }
    }
    if (currentMatch && currentMatch.players) {
      currentMatch.players.forEach(function (p) {
        if (p && p.pos !== myPos) opponents.push(p.name || ('Player' + p.pos));
      });
    }
    botEndRound(rank, opponents, matchId).then(function (resp) {
      if (resp && resp.session) {
        sessionPNL = resp.session;
        updateDashboard();
      }
    }).catch(function (err) {
      log('[BOT] End-round error: ' + err.message);
    });
  }

  function botHandleMatchSave(matchData) {
    if (!serverConnected) {
      log('[BOT] Server unreachable, saved to localStorage');
      return;
    }
    botSaveMatch(matchData).then(function () {
      log('[BOT] Saved to disk \u2713');
    }).catch(function (err) {
      log('[BOT] Save error: ' + err.message + ' (kept in localStorage)');
    });
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  SEND HELPERS (with seqNr management)
  // ═══════════════════════════════════════════════════════════════════════
  function sendBotMessage(msgId, data) {
    if (!gameSocket || gameSocket.readyState !== 1) {
      log('[BOT] Cannot send, WS not open');
      return false;
    }
    if (msgId < 9000) {
      data.seqNr = seqNr;
      seqNr++;
    }
    var payload = '4' + msgId + '#' + JSON.stringify(data);
    gameSocket.send(payload);
    return true;
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  MATCH / ROUND LIFECYCLE
  // ═══════════════════════════════════════════════════════════════════════
  function startMatch(data) {
    if (currentMatch && currentMatch.rounds.length > 0) {
      finalizeMatch();
    }

    const matchId = data.t_id || 'unknown';
    const rm = roomInfo[data.mt_id] || {};

    currentMatch = {
      match_id: matchId,
      timestamp: new Date().toISOString(),
      room: {
        id: data.mt_id,
        title: data.title || rm.title || '?',
        stake: data.bet || rm.stake || 0,
        prize_pool: data.pot || rm.pot || 0,
      },
      players: [],
      max_rounds: data.maxR || 4,
      rounds: [],
      final_standings: null,
      complete: false,
    };
    maxRounds = currentMatch.max_rounds;

    const pl = data.players || data.p;
    if (pl && Array.isArray(pl)) {
      currentMatch.players = pl.map(p => ({
        name: p.name || p.n || '?',
        uid: p.uid || null,
        pos: p.pos !== undefined ? p.pos : null,
        pic: p.pic || null,
      }));
    }

    log('=== MATCH: ' + matchId + ' | ' + currentMatch.room.title + ' | ' +
        currentMatch.room.stake + '/' + currentMatch.room.prize_pool + ' ===');
    log('Players: ' + currentMatch.players.map(p => p.name).join(', '));
    updateDashboard();
  }

  function startRound(data) {
    matchRound = data.currR || 1;
    trumpSuit = data.trump || null;

    currentRound = {
      round_number: matchRound,
      trump: trumpSuit,
      trump_id: SUIT_ID[trumpSuit] !== undefined ? SUIT_ID[trumpSuit] : null,
      trump_name: SUIT_NAME[trumpSuit] || trumpSuit,
      dealer: data.dealer !== undefined ? data.dealer : null,
      card_set: data.cardSet || 'FR',
      tricks: [],
      declarations: {},
      results: null,
      player_points: {},
      total_points: 0,
      joined_at_trick: null,
      my_hand: [],
      my_hand_ids: [],
    };

    trickNumber = 0;
    currentTrick = null;
    cardsInTrick = 0;
    isInGame = true;
    lastCompletedTrick = null;
    roundScores = { 0: 0, 1: 0, 2: 0, 3: 0 };
    clearTimeout(matchEndTimer);

    if (data.trick && data.trick > 0) {
      trickNumber = data.trick;
      currentRound.joined_at_trick = data.trick;
    }

    // seqNr sync from server
    if (data.seq !== undefined) {
      seqNr = data.seq;
      log('[SEQ] Synced seqNr to ' + seqNr);
    }

    // Track my position
    if (data.pos !== undefined && data.pos >= 0) {
      myPos = data.pos;
    }
    // Fallback: derive from players list (we're the non-bot player)
    if (myPos < 0 && currentMatch && currentMatch.players.length === 4) {
      for (var pi = 0; pi < 4; pi++) {
        var pl = currentMatch.players[pi];
        if (pl.uid === null || (pl.name && pl.name.indexOf('Gast') === 0)) {
          myPos = pi;
          break;
        }
      }
    }

    const stats = loadStats();
    if (trumpSuit && SUIT_ID[trumpSuit] !== undefined) stats.trumpDist[SUIT_ID[trumpSuit]]++;
    saveStats(stats);

    log('Round ' + matchRound + '/' + maxRounds + ' | Trump: ' + (SUIT_SYM[trumpSuit] || trumpSuit) +
        (currentRound.joined_at_trick ? ' | Joined trick ' + currentRound.joined_at_trick : ''));
    updateDashboard();
  }

  function finishRound() {
    if (!currentRound || !currentMatch) return;

    const pp = { 0: 0, 1: 0, 2: 0, 3: 0 };
    let totalPts = 0;
    currentRound.tricks.forEach(t => {
      if (t.winner !== null) pp[t.winner] = (pp[t.winner] || 0) + t.points;
      totalPts += t.points;
    });
    if (currentRound.tricks.length > 0) {
      const last = currentRound.tricks[currentRound.tricks.length - 1];
      if (last.trick_number >= 8 && last.winner !== null) {
        pp[last.winner] += 5;
        totalPts += 5;
      }
    }
    currentRound.player_points = pp;
    currentRound.total_points = totalPts;

    if (currentRound.results && currentRound.results.results) {
      currentRound.results.results.forEach(r => {
        if (r.callP !== undefined) currentRound.declarations[r.pos] = r.callP;
      });
    }

    currentMatch.rounds.push(currentRound);

    const stats = loadStats();
    stats.rounds++;
    if (currentRound.tricks.length === 9) stats.fullRounds++;

    if (currentRound.results && currentRound.results.results) {
      let hasPerfect = false;
      currentRound.results.results.forEach(r => {
        const dev = Math.abs((r.callP || 0) - (r.points || 0));
        if (dev === 0) hasPerfect = true;
        stats.totalDev += dev;
      });
      if (hasPerfect) stats.perfectRounds = (stats.perfectRounds || 0) + 1;

      const avgDev = currentRound.results.results.reduce(
        (s, r) => s + Math.abs((r.callP || 0) - (r.points || 0)), 0) / 4;
      stats.devHist.push(Math.round(avgDev));
      if (stats.devHist.length > 200) stats.devHist = stats.devHist.slice(-200);
      stats.pointsHist.push(totalPts);
      if (stats.pointsHist.length > 200) stats.pointsHist = stats.pointsHist.slice(-200);
    }

    if (currentMatch.room && currentMatch.room.title) {
      if (!stats.roomDist) stats.roomDist = {};
      stats.roomDist[currentMatch.room.title] = (stats.roomDist[currentMatch.room.title] || 0) + 1;
    }

    saveStats(stats);
    log('=== R' + matchRound + ' saved (' + currentRound.tricks.length + ' tricks, ' + totalPts + 'pts) ===');

    // Bot: notify round end
    botHandleRoundEnd();

    currentRound = null;
    currentTrick = null;
    isInGame = false;

    if (matchRound >= maxRounds) {
      finalizeMatch();
      log('Match complete. Cycling in ' + (CYCLE_DELAY / 1000) + 's...');
      if (!isPaused && !botPlayEnabled) scheduleAutoJoin(CYCLE_DELAY);
    } else if (botPlayEnabled) {
      // Bot mode: don't wait for next round — finalize immediately and rejoin
      log('[BOT] Round done, fast-cycling to next game...');
      finalizeMatch();
    } else {
      matchEndTimer = setTimeout(() => {
        log('No next round. Match ended early.');
        finalizeMatch();
        if (!isPaused) scheduleAutoJoin(1000);
      }, MATCH_END_TIMEOUT);
    }

    checkStorageCapacity();
    updateDashboard();
  }

  function finalizeMatch() {
    if (!currentMatch) return;
    currentMatch.complete = currentMatch.rounds.length >= maxRounds;

    if (currentMatch.rounds.length > 0) {
      const lastRound = currentMatch.rounds[currentMatch.rounds.length - 1];
      const standings = {};
      if (lastRound.results && lastRound.results.results) {
        lastRound.results.results.forEach(res => {
          standings[res.pos] = {
            total_deviation: Math.abs(res.total || 0),
            rounds_played: currentMatch.rounds.length,
            last_rank: res.rank,
          };
        });
      }
      currentMatch.final_standings = standings;
    }

    const matches = loadMatches();
    matches.push(currentMatch);
    saveMatches(matches);

    const stats = loadStats();
    stats.matches++;
    saveStats(stats);

    log('=== MATCH SAVED (#' + matches.length + ', ' + currentMatch.rounds.length + ' rounds) ===');

    // Bot: save match to disk
    botHandleMatchSave(currentMatch);

    // Bot: auto-join next training game (immediate for bot games)
    if (botPlayEnabled && serverConnected && !isPaused) {
      log('[BOT] Auto-joining next training game...');
      setTimeout(function () {
        if (gameSocket && gameSocket.readyState === 1) {
          gameSocket.send('49029#' + JSON.stringify({ mt_id: -1 }));
        }
      }, 500);
    }

    currentMatch = null;
    updateDashboard();
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  AUTO-JOIN
  // ═══════════════════════════════════════════════════════════════════════
  let gameSocket = null;

  function sendWS(msgId, data) {
    if (!gameSocket || gameSocket.readyState !== 1) return false;
    gameSocket.send('4' + msgId + '#' + (data !== undefined ? JSON.stringify(data) : ''));
    return true;
  }

  function joinTable(info) {
    if (!info || !info.t_id || !info.mt_id) return false;
    log('Joining table ' + info.t_id + '...');
    return sendWS('9033', { t_id: parseInt(info.t_id), mt_id: info.mt_id });
  }

  function scheduleAutoJoin(delay) {
    if (!autoJoinEnabled || isPaused) return;
    clearTimeout(autoJoinTimer);
    autoJoinTimer = setTimeout(() => {
      if (isInGame || isPaused) return;
      const t = pickNextTable();
      if (t) joinTable(t);
      else { log('No tables. Retry 10s...'); scheduleAutoJoin(10000); }
    }, delay || CYCLE_DELAY);
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  MESSAGE HANDLERS
  // ═══════════════════════════════════════════════════════════════════════
  const seenMsgIds = new Set();

  function handleMessage(msgId, body) {
    if (!seenMsgIds.has(msgId)) { seenMsgIds.add(msgId); log('New msg: ' + msgId, body); }

    switch (msgId) {
      case '9022': onGameInit(body); break;
      case '9011':
        if (body && body.ar) body.ar.forEach(sub => {
          if (typeof sub === 'string') parseAndDispatch(sub);
        });
        break;
      case '9021': onTableUpdate(body); break;
      case '9035': onRoomList(body); break;
      case '1': onCardsDealt(body); break;
      case '2': break;
      case '3': break;
      case '4': break;
      case '5': onPlayableCards(body); break;
      case '6': onCardPlayed(body); break;
      case '7': onCardPlayed(body); break;
      case '8': onTrickWon(body); break;
      case '9': onRemoveTrick(body); break;
      case '10': onResult(body); break;
      case '17': onFinalResult(body); break;
      case '19': onTrumpMsg(body); break;
      case '9001': case '9002': case '9005': case '9024': case '9025':
      case '9034': case '9046': case '9051': case '9052': case '9058': case '9060': break;
      default:
        if (isInGame && parseInt(msgId) < 100) log('Msg ' + msgId, body);
        break;
    }
  }

  function onGameInit(data) {
    if (!data) return;
    clearTimeout(matchEndTimer);
    const matchId = data.t_id || null;
    const isNewMatch = !currentMatch || currentMatch.match_id !== matchId;
    if (isNewMatch) startMatch(data);
    startRound(data);

    // Handle hand + declaration for bot play
    if (data.cards && Array.isArray(data.cards)) {
      myHand = data.cards.slice();
      currentRound.my_hand = data.cards.slice();
      currentRound.my_hand_ids = data.cards.map(c => {
        const p = parseCard(c); return p ? p.id : c;
      });
      log('Hand: ' + data.cards.map(cardDisplay).join(' '));

      if (data.selectScreen && botPlayEnabled && serverConnected) {
        botHandleDeclaration(myHand.slice(), trumpSuit);
      }
    }
  }

  function onCardsDealt(data) {
    if (!currentRound) return;
    if (data && data.cards && Array.isArray(data.cards)) {
      myHand = data.cards.slice();
      currentRound.my_hand = data.cards.slice();
      currentRound.my_hand_ids = data.cards.map(c => {
        const p = parseCard(c); return p ? p.id : c;
      });
      log('Hand: ' + data.cards.map(cardDisplay).join(' '));
    }
  }

  function onPlayableCards(body) {
    if (!currentRound || !body) return;
    var pCards = body.pCards;
    if (!pCards || !Array.isArray(pCards) || pCards.length === 0) return;
    log('Playable: ' + pCards.map(cardDisplay).join(' '));

    if (botPlayEnabled && serverConnected) {
      botHandlePlayCard(pCards);
    }
  }

  function onTrumpMsg(data) {
    if (!currentRound || !data) return;
    const t = data.trump || data.t;
    if (t && SUIT_ID[t] !== undefined) {
      trumpSuit = t;
      currentRound.trump = t;
      currentRound.trump_id = SUIT_ID[t];
      log('Trump changed: ' + SUIT_SYM[t]);
    }
  }

  function onCardPlayed(data) {
    if (!currentRound || !data) return;
    const pos = data.pos !== undefined ? data.pos : cardsInTrick;
    const code = data.card || data.c;
    if (!code) return;

    if (!currentTrick) {
      currentTrick = {
        trick_number: trickNumber, leader: pos,
        cards: [], winner: null, points: 0,
      };
      cardsInTrick = 0;
    }
    if (currentTrick.cards.some(c => c.card === code && c.player === pos)) return;

    const pts = cardPoints(code, trumpSuit);
    const parsed = parseCard(code);
    currentTrick.cards.push({
      player: pos, card: code,
      card_id: parsed ? parsed.id : null,
      points: pts,
      is_trump: code.charAt(0) === trumpSuit,
    });
    cardsInTrick++;
    updateDashboard();
  }

  function onTrickWon(data) {
    if (!currentRound) return;
    const w = data && data.pos !== undefined ? data.pos : null;
    if (currentTrick) {
      currentTrick.winner = w;
      currentTrick.points = currentTrick.cards.reduce((s, c) => s + (c.points || 0), 0);
      currentRound.tricks.push(currentTrick);
      if (w !== null) roundScores[w] = (roundScores[w] || 0) + currentTrick.points;
      lastCompletedTrick = currentTrick;
      log('Trick ' + trickNumber + ': ' +
          currentTrick.cards.map(c => cardDisplay(c.card)).join(' ') +
          ' \u2192 seat ' + w + ' (' + currentTrick.points + 'pts)');

      // Bot: notify trick
      var trickPts = data && data.points !== undefined ? data.points : currentTrick.points;
      botHandleTrickDone(currentTrick, w, trickPts);
    }
    currentTrick = null;
    cardsInTrick = 0;
    trickNumber++;
    updateDashboard();
  }

  function onRemoveTrick(data) {
    if (currentTrick && currentTrick.cards.length >= 4) onTrickWon(data || {});
  }

  function onResult(data) {
    if (!currentRound) return;
    log('Result', data);
    currentRound.results = data;
    finishRound();
  }

  function onFinalResult(data) {
    if (!currentRound) return;
    log('Final', data);
    if (!currentRound.results) currentRound.results = data;
    matchRound = maxRounds;
    finishRound();
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  EIO v4 PARSING
  // ═══════════════════════════════════════════════════════════════════════
  function parseAndDispatch(raw) {
    if (typeof raw !== 'string' || !raw.length) return;
    const c = raw.charAt(0);
    if ('0123'.includes(c)) return;
    const p = c === '4' ? raw.substring(1) : raw;
    const h = p.indexOf('#');
    if (h < 0) return;
    const id = p.substring(0, h), js = p.substring(h + 1);
    if (!/^\d+$/.test(id)) return;
    let b;
    try { b = js ? JSON.parse(js) : {}; } catch { b = js; }
    handleMessage(id, b);
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  WEBSOCKET PATCH  (must execute at document-start)
  // ═══════════════════════════════════════════════════════════════════════
  const OrigWS = window.WebSocket;
  window.WebSocket = function (url, proto) {
    const ws = proto ? new OrigWS(url, proto) : new OrigWS(url);
    if (url && url.includes('gs.swisslos.ch')) {
      gameSocket = ws;
      logQueue.push({ time: new Date().toLocaleTimeString(), text: '[WS] ' + url, isMatch: false });
      ws.addEventListener('message', e => {
        if (typeof e.data === 'string') parseAndDispatch(e.data);
      });
      ws.addEventListener('open', () => {
        log('[WS] Open');
        // On page load: if bot mode was active, auto-join training game
        setTimeout(() => {
          if (botPlayEnabled && gameSocket && gameSocket.readyState === 1) {
            log('[BOT] Resuming bot play mode (persisted)');
            gameSocket.send('49029#' + JSON.stringify({ mt_id: -1 }));
          } else {
            scheduleAutoJoin(AUTO_JOIN_DELAY);
          }
        }, 2000);
      });
      ws.addEventListener('close', () => log('[WS] Closed'));
    }
    return ws;
  };
  window.WebSocket.prototype = OrigWS.prototype;
  window.WebSocket.CONNECTING = OrigWS.CONNECTING;
  window.WebSocket.OPEN = OrigWS.OPEN;
  window.WebSocket.CLOSING = OrigWS.CLOSING;
  window.WebSocket.CLOSED = OrigWS.CLOSED;

  // Patch send to log outgoing messages
  const origSend = OrigWS.prototype.send;
  OrigWS.prototype.send = function (data) {
    if (this === gameSocket && typeof data === 'string' && data !== '3' && !data.startsWith('3')) {
      // Skip ping (EIO '3'), log everything else
      const preview = data.length > 120 ? data.substring(0, 120) + '...' : data;
      log('>> ' + preview);
    }
    return origSend.apply(this, arguments);
  };

  // ═══════════════════════════════════════════════════════════════════════
  //  LOGGING
  // ═══════════════════════════════════════════════════════════════════════
  let logQueue = [];
  let logEntries = [];
  const MAX_LOG = 500;

  function log(msg, data) {
    const time = new Date().toLocaleTimeString();
    const text = data !== undefined
      ? msg + ' ' + JSON.stringify(data).substring(0, 200)
      : msg;
    console.log('[JC:Tab' + tabNumber + '] ' + text);
    const entry = { time: time, text: text, isMatch: msg.startsWith('===') };
    logEntries.push(entry);
    if (logEntries.length > MAX_LOG) logEntries.shift();
    if (!dashboardReady) { logQueue.push(entry); return; }
    appendLogEntry(entry);
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  EXPORT
  // ═══════════════════════════════════════════════════════════════════════
  function buildExport() {
    const matches = loadMatches();
    const stats = loadStats();
    return {
      exported_at: new Date().toISOString(),
      collector_version: '7.0',
      total_matches: matches.length,
      total_rounds: stats.rounds,
      full_rounds: stats.fullRounds,
      collection_hours: ((Date.now() - stats.started) / 3600000).toFixed(2),
      card_encoding: {
        suits: { H: 0, D: 1, S: 2, C: 3 },
        values: { '6': 0, '7': 1, '8': 2, '9': 3, '10': 4, J: 5, Q: 6, K: 7, A: 8 },
        formula: 'card_id = suit * 9 + value',
        total_cards: 36,
        base_points: { '6': 0, '7': 0, '8': 0, '9': 0, '10': 10, J: 2, Q: 3, K: 4, A: 11 },
        trump_points: { '6': 0, '7': 0, '8': 0, '9': 14, '10': 10, J: 20, Q: 3, K: 4, A: 11 },
        round_total: 157,
      },
      trump_distribution: {
        herz: stats.trumpDist[0], ecke: stats.trumpDist[1],
        schaufel: stats.trumpDist[2], kreuz: stats.trumpDist[3],
      },
      matches: matches,
    };
  }

  function exportJSON() {
    const m = loadMatches();
    if (!m.length) { log('No data to export.'); return; }
    const blob = new Blob([JSON.stringify(buildExport(), null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'jass_data_' + new Date().toISOString().replace(/[:.]/g, '-') + '.json';
    a.click();
    log('Exported ' + m.length + ' matches');
  }

  function copyClip() {
    const m = loadMatches();
    if (!m.length) { log('No data to copy.'); return; }
    navigator.clipboard.writeText(JSON.stringify(buildExport(), null, 2)).then(() => {
      log('Copied to clipboard');
    }).catch(() => {
      const ta = document.createElement('textarea');
      ta.value = JSON.stringify(buildExport(), null, 2);
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      log('Copied to clipboard (fallback)');
    });
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  DASHBOARD — CSS (Floating Overlay Panel)
  // ═══════════════════════════════════════════════════════════════════════
  let dashboardReady = false;
  let panelMinimized = false;
  let logWindowOpen = false;
  let dashEl = null;

  function getDashboardCSS() {
    return '' +
    '#jc-panel{position:fixed;bottom:12px;right:12px;width:380px;max-height:85vh;' +
    'background:#0d0e14;border:1px solid #2a2d3a;border-radius:10px;' +
    'box-shadow:0 4px 24px rgba(0,0,0,.6);z-index:2147483647;' +
    'font-family:Consolas,Monaco,"Courier New",monospace;font-size:12px;color:#ccc;' +
    'display:flex;flex-direction:column;overflow:hidden}' +
    '#jc-panel.minimized{max-height:none;height:auto}' +
    '#jc-panel.minimized #jc-panel-body{display:none}' +
    '#jc-panel.minimized #jc-panel-footer{display:none}' +

    /* header bar */
    '#jc-panel-header{background:#12131a;padding:8px 12px;display:flex;align-items:center;gap:8px;' +
    'cursor:grab;user-select:none;border-bottom:1px solid #2a2d3a;flex-shrink:0}' +
    '#jc-panel-header:active{cursor:grabbing}' +
    '#jc-panel-header .logo{font-size:13px;font-weight:700;color:#6dd5fa;white-space:nowrap}' +
    '#jc-panel-header .status-dot{width:7px;height:7px;border-radius:50%;background:#555;flex-shrink:0}' +
    '#jc-panel-header .status-dot.on{background:#4caf50;box-shadow:0 0 6px #4caf50}' +
    '#jc-panel-header .status-text{color:#666;font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}' +
    '#jc-panel-header .tab-input{width:30px;padding:1px 4px;border:1px solid #2a2d3a;border-radius:3px;' +
    'background:#1a1c2e;color:#fff;font-family:inherit;font-size:11px;text-align:center}' +
    '#jc-panel-header .hdr-right{display:flex;align-items:center;gap:6px;margin-left:auto;flex-shrink:0}' +
    '#jc-panel-header .minimize-btn{background:none;border:1px solid #2a2d3a;border-radius:3px;' +
    'color:#999;cursor:pointer;font-size:14px;line-height:1;padding:1px 6px;font-family:inherit}' +
    '#jc-panel-header .minimize-btn:hover{color:#fff;border-color:#555}' +

    /* mode toggle */
    '.jc-mode-toggle{display:flex;align-items:center;gap:6px;padding:6px 8px;background:#12131a;' +
    'border-bottom:1px solid #2a2d3a;flex-shrink:0}' +
    '.jc-mode-btn{padding:4px 12px;border:1px solid #2a2d3a;border-radius:4px;' +
    'background:#1a1c2e;color:#999;cursor:pointer;font-family:inherit;font-size:11px;' +
    'font-weight:700;transition:all .15s}' +
    '.jc-mode-btn:hover{border-color:#555;color:#ccc}' +
    '.jc-mode-btn.active-watch{background:#1b2a3a;border-color:#369;color:#6dd5fa}' +
    '.jc-mode-btn.active-bot{background:#1b3a1b;border-color:#3a6;color:#6d6}' +
    '.jc-server-status{font-size:9px;color:#666;margin-left:4px}' +
    '.jc-server-status.connected{color:#4caf50}' +
    '.jc-server-status.disconnected{color:#ef5350}' +
    '.jc-timing-display{font-size:9px;color:#888;margin-left:auto}' +

    /* scrollable body */
    '#jc-panel-body{overflow-y:auto;overflow-x:hidden;padding:8px;display:flex;flex-direction:column;gap:8px;' +
    'flex:1;min-height:0}' +
    '#jc-panel-body::-webkit-scrollbar{width:4px}' +
    '#jc-panel-body::-webkit-scrollbar-thumb{background:#2a2d3a;border-radius:2px}' +

    /* generic section */
    '.jc-section{background:#1a1c2e;border:1px solid #2a2d3a;border-radius:6px;padding:8px}' +
    '.jc-section-title{color:#6dd5fa;font-size:10px;text-transform:uppercase;' +
    'letter-spacing:.5px;margin-bottom:6px;font-weight:600}' +

    /* stats grid */
    '.jc-stat-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px}' +
    '.jc-stat .lbl{color:#666;font-size:9px}' +
    '.jc-stat .val{color:#fff;font-size:13px;font-weight:700}' +
    '.jc-stat .val.green{color:#4caf50}.jc-stat .val.orange{color:#ffa726}' +
    '.jc-stat .val.red{color:#ef5350}.jc-stat .val.blue{color:#6dd5fa}' +

    /* match info */
    '.jc-match-info{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:6px}' +
    '.jc-match-info .item{display:flex;flex-direction:column}' +
    '.jc-match-info .item .lbl{color:#666;font-size:8px;text-transform:uppercase}' +
    '.jc-match-info .item .val{color:#fff;font-size:11px;font-weight:600}' +
    '.jc-players-row{font-size:10px;color:#999;margin-bottom:4px;word-break:break-word}' +
    '.jc-round-info{display:flex;gap:12px;align-items:center;flex-wrap:wrap}' +
    '.jc-trump-display{font-size:18px;display:flex;align-items:center;gap:4px}' +
    '.jc-trump-display .name{font-size:10px;color:#999}' +

    /* deviation bars (current match) */
    '.jc-dev-bars{display:flex;gap:6px;margin-top:6px;align-items:flex-end}' +
    '.jc-dev-bar-item{display:flex;flex-direction:column;align-items:center;gap:1px}' +
    '.jc-dev-bar-item .bar{width:22px;border-radius:2px 2px 0 0;transition:height .3s;min-height:2px}' +
    '.jc-dev-bar-item .label{font-size:8px;color:#666}' +
    '.jc-dev-bar-item .value{font-size:9px;color:#fff;font-weight:600}' +

    /* live trick */
    '.jc-trick-live{display:flex;gap:6px;align-items:center;padding:4px 0;font-size:14px;flex-wrap:wrap}' +
    '.jc-trick-card{padding:3px 6px;background:#0d0e14;border:1px solid #2a2d3a;' +
    'border-radius:3px;min-width:38px;text-align:center;font-weight:600;font-size:13px}' +
    '.jc-trick-card.empty{color:#333;border-style:dashed}' +
    '.jc-trick-card .seat{font-size:7px;color:#555;display:block}' +
    '.jc-last-trick{font-size:10px;color:#777;margin-top:3px}' +
    '.jc-last-trick .winner{color:#4caf50;font-weight:600}' +
    '.jc-running-scores{display:flex;gap:10px;margin-top:4px;font-size:10px}' +
    '.jc-running-scores .player-score{display:flex;flex-direction:column;align-items:center}' +
    '.jc-running-scores .player-score .name{color:#666;font-size:8px;max-width:55px;' +
    'overflow:hidden;text-overflow:ellipsis;white-space:nowrap}' +
    '.jc-running-scores .player-score .pts{color:#fff;font-weight:700;font-size:12px}' +

    /* bar rows (trump) */
    '.jc-bar-row{display:flex;align-items:center;gap:5px;margin:2px 0}' +
    '.jc-bar-label{width:16px;font-size:11px;text-align:center;flex-shrink:0}' +
    '.jc-bar-bg{flex:1;height:8px;background:#0d0e14;border-radius:4px;overflow:hidden}' +
    '.jc-bar-fill{height:100%;border-radius:4px;transition:width .3s;min-width:0}' +
    '.jc-bar-count{width:28px;text-align:right;font-size:9px;color:#666;flex-shrink:0}' +

    /* deviation chart (mini bar chart) */
    '.jc-dev-chart{display:flex;align-items:flex-end;gap:2px;height:40px}' +
    '.jc-dev-chart .bar{flex:1;border-radius:2px 2px 0 0;min-width:3px;max-width:14px;transition:height .3s}' +

    /* history strip */
    '.jc-history-strip{display:flex;gap:2px;flex-wrap:wrap}' +
    '.jc-history-block{width:14px;height:14px;border-radius:2px;cursor:default;' +
    'display:flex;align-items:center;justify-content:center;' +
    'font-size:7px;color:rgba(255,255,255,.5);font-weight:700}' +

    /* scatter plot - improved */
    '.jc-scatter{position:relative;width:100%;height:180px;background:#0d0e14;' +
    'border:1px solid #2a2d3a;border-radius:3px;overflow:hidden}' +
    '.jc-scatter .axis-label{position:absolute;font-size:8px;color:#555}' +
    '.jc-scatter .diagonal{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}' +
    '.jc-scatter .dot{position:absolute;width:6px;height:6px;border-radius:50%;transform:translate(-3px,-3px);opacity:0.7}' +
    '.jc-scatter .axis-line-h{position:absolute;left:0;right:0;height:1px;background:#1a1c2e}' +
    '.jc-scatter .axis-line-v{position:absolute;top:0;bottom:0;width:1px;background:#1a1c2e}' +
    '.jc-scatter .grid-label{position:absolute;font-size:7px;color:#444;pointer-events:none}' +
    '.jc-scatter .scatter-count{position:absolute;top:4px;right:6px;font-size:9px;color:#555}' +

    /* leaderboard */
    '.jc-lb-wrap{max-height:120px;overflow-y:auto}' +
    '.jc-lb-wrap::-webkit-scrollbar{width:3px}' +
    '.jc-lb-wrap::-webkit-scrollbar-thumb{background:#2a2d3a;border-radius:2px}' +
    '.jc-leaderboard{width:100%;border-collapse:collapse}' +
    '.jc-leaderboard th{text-align:left;color:#555;font-size:9px;text-transform:uppercase;' +
    'padding:3px 4px;border-bottom:1px solid #2a2d3a;font-weight:600;white-space:nowrap;position:sticky;top:0;background:#1a1c2e}' +
    '.jc-leaderboard td{padding:2px 4px;border-bottom:1px solid rgba(42,45,58,.5);font-size:10px;white-space:nowrap}' +
    '.jc-leaderboard tr:hover td{background:rgba(109,213,250,.04)}' +
    '.jc-leaderboard .rank{color:#555;font-weight:700;width:20px}' +
    '.jc-leaderboard .player-name{color:#ccc;max-width:80px;overflow:hidden;text-overflow:ellipsis}' +
    '.jc-leaderboard .num{text-align:right;color:#fff;font-weight:600}' +
    '.jc-leaderboard .num.green{color:#4caf50}' +
    '.jc-leaderboard .num.orange{color:#ffa726}' +
    '.jc-leaderboard .num.red{color:#ef5350}' +
    '.jc-sparkline{display:inline-flex;gap:1px;align-items:flex-end;height:10px;vertical-align:middle;margin-left:3px}' +
    '.jc-sparkline .bar{width:2px;border-radius:1px;min-height:1px}' +

    /* room list */
    '.jc-room-row{display:flex;justify-content:space-between;font-size:10px;padding:1px 0}' +
    '.jc-room-row .name{color:#999}.jc-room-row .count{color:#fff;font-weight:600}' +

    /* storage meter */
    '.jc-storage-meter{margin-top:3px}' +
    '.jc-storage-bg{height:5px;background:#0d0e14;border-radius:3px;overflow:hidden}' +
    '.jc-storage-fill{height:100%;border-radius:3px;background:#4caf50;transition:width .3s}' +
    '.jc-storage-fill.warn{background:#ffa726}.jc-storage-fill.critical{background:#ef5350}' +
    '.jc-storage-text{font-size:8px;color:#555;margin-top:1px}' +

    /* log window (separate floating) */
    '#jc-log-window{position:fixed;bottom:12px;right:400px;width:450px;max-height:500px;' +
    'background:#0d0e14;border:1px solid #2a2d3a;border-radius:10px;' +
    'box-shadow:0 4px 24px rgba(0,0,0,.6);z-index:2147483646;' +
    'font-family:Consolas,Monaco,"Courier New",monospace;font-size:12px;color:#ccc;' +
    'display:none;flex-direction:column;overflow:hidden}' +
    '#jc-log-window.open{display:flex}' +
    '#jc-log-window-header{background:#12131a;padding:8px 12px;display:flex;align-items:center;' +
    'justify-content:space-between;border-bottom:1px solid #2a2d3a;cursor:grab;user-select:none}' +
    '#jc-log-window-header:active{cursor:grabbing}' +
    '#jc-log-window-header .title{color:#6dd5fa;font-size:11px;font-weight:700}' +
    '#jc-log-window-header .close-btn{background:none;border:1px solid #2a2d3a;border-radius:3px;' +
    'color:#999;cursor:pointer;font-size:12px;line-height:1;padding:1px 6px;font-family:inherit}' +
    '#jc-log-window-header .close-btn:hover{color:#fff;border-color:#555}' +
    '.jc-log-area{flex:1;overflow-y:auto;padding:6px 8px;font-size:9px;line-height:1.4;min-height:0}' +
    '.jc-log-area::-webkit-scrollbar{width:3px}' +
    '.jc-log-area::-webkit-scrollbar-thumb{background:#2a2d3a;border-radius:2px}' +
    '.jc-log-entry{padding:1px 0}.jc-log-entry .time{color:#333;margin-right:4px}' +
    '.jc-log-entry.match{color:#4caf50;font-weight:700}' +

    /* footer buttons */
    '#jc-panel-footer{background:#12131a;border-top:1px solid #2a2d3a;padding:6px 8px;flex-shrink:0;' +
    'display:flex;flex-wrap:wrap;gap:4px;align-items:center}' +
    '.jc-btn{padding:3px 8px;border:1px solid #2a2d3a;border-radius:4px;' +
    'background:#1a1c2e;color:#999;cursor:pointer;font-family:inherit;font-size:10px;' +
    'white-space:nowrap;transition:all .15s}' +
    '.jc-btn:hover{background:#2a3050;color:#fff;border-color:#444}' +
    '.jc-btn.active{background:#1b3a1b;border-color:#3a6;color:#6d6}' +
    '.jc-btn.danger{color:#ef5350;font-size:9px;padding:2px 6px}.jc-btn.danger:hover{background:#3a1b1b;border-color:#a33}' +
    '.jc-btn.primary{color:#6dd5fa}.jc-btn.primary:hover{background:#1b2a3a;border-color:#369}' +

    '#jc-panel *{scrollbar-width:thin;scrollbar-color:#2a2d3a #0d0e14}' +
    '#jc-log-window *{scrollbar-width:thin;scrollbar-color:#2a2d3a #0d0e14}';
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  DASHBOARD — HTML (Floating Overlay Panel)
  // ═══════════════════════════════════════════════════════════════════════
  function getDashboardHTML() {
    return '' +
    /* HEADER */
    '<div id="jc-panel-header">' +
      '<span class="logo">JC v7.0</span>' +
      '<input type="number" class="tab-input" id="jc-tab-input" value="' + tabNumber + '" min="1" max="99" title="Tab number">' +
      '<div class="status-dot" id="jc-status-dot"></div>' +
      '<span class="status-text" id="jc-status-text">init</span>' +
      '<div class="hdr-right">' +
        '<span id="jc-storage-warn" style="color:#ffa726;font-size:9px;display:none"></span>' +
        '<button class="minimize-btn" id="jc-minimize-btn" title="Minimize/Expand">&ndash;</button>' +
      '</div>' +
    '</div>' +

    /* MODE TOGGLE BAR */
    '<div class="jc-mode-toggle" id="jc-mode-toggle">' +
      '<button class="jc-mode-btn active-watch" id="jc-mode-watch">WATCH</button>' +
      '<button class="jc-mode-btn" id="jc-mode-assist">ASSIST</button>' +
      '<button class="jc-mode-btn" id="jc-mode-bot">BOT PLAY</button>' +
      '<span class="jc-server-status disconnected" id="jc-server-status">Server: Disconnected</span>' +
      '<span class="jc-timing-display" id="jc-timing-display"></span>' +
    '</div>' +

    /* PNL Display */
    '<div id="jc-pnl-display" style="padding:4px 10px;font-size:10px;color:#aaa;' +
      'background:#0a0e14;border-bottom:1px solid #222;font-family:monospace;">' +
      'SESSION: connecting...' +
    '</div>' +

    /* BODY */
    '<div id="jc-panel-body">' +

      /* Stats Grid */
      '<div class="jc-section">' +
        '<div class="jc-section-title">Stats</div>' +
        '<div class="jc-stat-grid">' +
          '<div class="jc-stat"><div class="lbl">Matches</div><div class="val blue" id="jc-s-matches">0</div></div>' +
          '<div class="jc-stat"><div class="lbl">Rounds</div><div class="val" id="jc-s-rounds">0</div></div>' +
          '<div class="jc-stat"><div class="lbl">Complete</div><div class="val green" id="jc-s-complete">0</div></div>' +
          '<div class="jc-stat"><div class="lbl">Rate</div><div class="val" id="jc-s-rate">-</div></div>' +
          '<div class="jc-stat"><div class="lbl">Avg Dev</div><div class="val" id="jc-s-avgdev">-</div></div>' +
          '<div class="jc-stat"><div class="lbl">Perfect %</div><div class="val green" id="jc-s-perfectpct">-</div></div>' +
        '</div>' +
      '</div>' +

      /* Current Match */
      '<div class="jc-section" id="jc-current-match">' +
        '<div class="jc-section-title">Current Match</div>' +
        '<div id="jc-match-content"><div style="color:#555;font-size:11px">Waiting for game data...</div></div>' +
      '</div>' +

      /* Live Trick */
      '<div class="jc-section" id="jc-live-game">' +
        '<div class="jc-section-title">Live Trick</div>' +
        '<div id="jc-live-content"><div style="color:#555;font-size:11px">No active round.</div></div>' +
      '</div>' +

      /* Trump Distribution */
      '<div class="jc-section">' +
        '<div class="jc-section-title">Trump Distribution</div>' +
        '<div id="jc-trump-bars"></div>' +
      '</div>' +

      /* Deviation Chart (last 20 rounds) */
      '<div class="jc-section">' +
        '<div class="jc-section-title">Deviation (last 20 rounds)</div>' +
        '<div class="jc-dev-chart" id="jc-dev-chart"></div>' +
      '</div>' +

      /* Match History Strip */
      '<div class="jc-section">' +
        '<div class="jc-section-title">Match History (last 20)</div>' +
        '<div class="jc-history-strip" id="jc-history-strip"></div>' +
      '</div>' +

      /* Declaration Scatter */
      '<div class="jc-section">' +
        '<div class="jc-section-title">Declaration Scatter</div>' +
        '<div class="jc-scatter" id="jc-scatter"></div>' +
      '</div>' +

      /* Player Leaderboard */
      '<div class="jc-section">' +
        '<div class="jc-section-title">Player Leaderboard</div>' +
        '<div class="jc-lb-wrap">' +
          '<table class="jc-leaderboard" id="jc-leaderboard">' +
            '<thead><tr>' +
              '<th>#</th><th>Name</th><th>Rds</th><th>AvgD</th>' +
              '<th>Best</th><th>Perf</th><th>Trend</th>' +
            '</tr></thead>' +
            '<tbody id="jc-lb-body"></tbody>' +
          '</table>' +
        '</div>' +
      '</div>' +

      /* Room Distribution */
      '<div class="jc-section">' +
        '<div class="jc-section-title">Rooms</div>' +
        '<div id="jc-room-dist"></div>' +
      '</div>' +

      /* Storage Meter */
      '<div class="jc-section">' +
        '<div class="jc-section-title">Storage</div>' +
        '<div class="jc-storage-meter">' +
          '<div class="jc-storage-bg"><div class="jc-storage-fill" id="jc-storage-fill"></div></div>' +
          '<div class="jc-storage-text" id="jc-storage-text">0 KB / ~5 MB</div>' +
        '</div>' +
      '</div>' +

    '</div>' +

    /* FOOTER (sticky buttons) */
    '<div id="jc-panel-footer">' +
      '<button class="jc-btn primary" id="jc-btn-export">Export</button>' +
      '<button class="jc-btn" id="jc-btn-copy">Copy</button>' +
      '<button class="jc-btn active" id="jc-btn-auto">Auto ON</button>' +
      '<button class="jc-btn" id="jc-btn-join">Join</button>' +
      '<button class="jc-btn" id="jc-btn-pause">Pause</button>' +
      '<button class="jc-btn" id="jc-btn-log">Log</button>' +
      '<button class="jc-btn danger" id="jc-btn-clear">Clear</button>' +
    '</div>';
  }

  function getLogWindowHTML() {
    return '' +
    '<div id="jc-log-window-header">' +
      '<span class="title">Log</span>' +
      '<button class="close-btn" id="jc-log-close-btn">&times;</button>' +
    '</div>' +
    '<div class="jc-log-area" id="jc-log-area"></div>';
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  DASHBOARD — CREATE & WIRE (Floating Panel)
  // ═══════════════════════════════════════════════════════════════════════
  let logWindowEl = null;

  function createDashboard() {
    // Just append a floating panel -- do NOT touch the original page DOM
    dashEl = document.createElement('div');
    dashEl.id = 'jc-panel';
    document.body.appendChild(dashEl);

    // Create separate log window
    logWindowEl = document.createElement('div');
    logWindowEl.id = 'jc-log-window';
    document.body.appendChild(logWindowEl);

    // Inject CSS
    var style = document.createElement('style');
    style.textContent = getDashboardCSS();
    document.head.appendChild(style);

    // Populate HTML
    dashEl.innerHTML = getDashboardHTML();
    logWindowEl.innerHTML = getLogWindowHTML();

    // Wire interactive elements
    wireEvents();

    // Make draggable by header
    makeDraggable(dashEl, document.getElementById('jc-panel-header'));
    makeDraggable(logWindowEl, document.getElementById('jc-log-window-header'));

    dashboardReady = true;

    // Flush log queue
    logQueue.forEach(function (entry) { appendLogEntry(entry); });
    logQueue = [];

    updateTitle();
    updateDashboard();
  }

  function makeDraggable(panel, handle) {
    var isDragging = false, startX, startY, startLeft, startTop;
    handle.addEventListener('mousedown', function (e) {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON') return;
      isDragging = true;
      var rect = panel.getBoundingClientRect();
      startX = e.clientX;
      startY = e.clientY;
      startLeft = rect.left;
      startTop = rect.top;
      e.preventDefault();
    });
    document.addEventListener('mousemove', function (e) {
      if (!isDragging) return;
      var dx = e.clientX - startX;
      var dy = e.clientY - startY;
      panel.style.left = (startLeft + dx) + 'px';
      panel.style.top = (startTop + dy) + 'px';
      panel.style.right = 'auto';
      panel.style.bottom = 'auto';
    });
    document.addEventListener('mouseup', function () {
      isDragging = false;
    });
  }

  function wireEvents() {
    document.getElementById('jc-tab-input').addEventListener('change', function () {
      tabNumber = parseInt(this.value) || 1;
      sessionStorage.setItem('jc_tab_number', tabNumber);
      updateTitle();
    });

    document.getElementById('jc-minimize-btn').addEventListener('click', function () {
      panelMinimized = !panelMinimized;
      dashEl.classList.toggle('minimized', panelMinimized);
      this.innerHTML = panelMinimized ? '&#9634;' : '&ndash;';
      this.title = panelMinimized ? 'Expand' : 'Minimize';
    });

    // Mode toggle — helper to reset all mode buttons
    function setMode(mode) {
      botPlayEnabled = mode === 'bot';
      assistantMode = mode === 'assist';
      localStorage.setItem('jc_bot_mode', botPlayEnabled ? 'true' : 'false');
      localStorage.setItem('jc_assistant_mode', assistantMode ? 'true' : 'false');
      document.getElementById('jc-mode-watch').className = 'jc-mode-btn' + (mode === 'watch' ? ' active-watch' : '');
      document.getElementById('jc-mode-assist').className = 'jc-mode-btn' + (mode === 'assist' ? ' active-bot' : '');
      document.getElementById('jc-mode-bot').className = 'jc-mode-btn' + (mode === 'bot' ? ' active-bot' : '');
      log('[MODE] Switched to ' + mode.toUpperCase());
      updateDashboard();
    }

    document.getElementById('jc-mode-watch').addEventListener('click', function () {
      setMode('watch');
    });

    document.getElementById('jc-mode-assist').addEventListener('click', function () {
      if (!serverConnected) {
        log('[ASSIST] Cannot enable: server disconnected');
        return;
      }
      setMode('assist');
    });

    document.getElementById('jc-mode-bot').addEventListener('click', function () {
      if (!serverConnected) {
        log('[BOT] Cannot enable Bot Play: server disconnected');
        return;
      }
      setMode('bot');
    });

    // Log window toggle
    document.getElementById('jc-btn-log').addEventListener('click', function () {
      logWindowOpen = !logWindowOpen;
      logWindowEl.classList.toggle('open', logWindowOpen);
      this.className = 'jc-btn' + (logWindowOpen ? ' active' : '');
      if (logWindowOpen) {
        // Position to the left of main panel
        var panelRect = dashEl.getBoundingClientRect();
        logWindowEl.style.right = (window.innerWidth - panelRect.left + 8) + 'px';
        logWindowEl.style.bottom = panelRect.bottom > window.innerHeight
          ? '12px'
          : (window.innerHeight - panelRect.bottom) + 'px';
        logWindowEl.style.left = 'auto';
        logWindowEl.style.top = 'auto';
        // Scroll to bottom
        var area = document.getElementById('jc-log-area');
        if (area) area.scrollTop = area.scrollHeight;
      }
    });

    document.getElementById('jc-log-close-btn').addEventListener('click', function () {
      logWindowOpen = false;
      logWindowEl.classList.remove('open');
      var logBtn = document.getElementById('jc-btn-log');
      if (logBtn) logBtn.className = 'jc-btn';
    });

    document.getElementById('jc-btn-export').addEventListener('click', exportJSON);
    document.getElementById('jc-btn-copy').addEventListener('click', copyClip);

    document.getElementById('jc-btn-clear').addEventListener('click', function () {
      if (!confirm('Delete all collected data? This cannot be undone.')) return;
      saveMatches([]);
      saveStats(defaultStats());
      allPlayers = {};
      log('All data cleared.');
      updateDashboard();
    });

    document.getElementById('jc-btn-auto').addEventListener('click', function () {
      autoJoinEnabled = !autoJoinEnabled;
      this.textContent = 'Auto ' + (autoJoinEnabled ? 'ON' : 'OFF');
      this.className = 'jc-btn' + (autoJoinEnabled ? ' active' : '');
      if (autoJoinEnabled && !isInGame && !isPaused) scheduleAutoJoin(2000);
    });

    document.getElementById('jc-btn-join').addEventListener('click', function () {
      if (botPlayEnabled && serverConnected) {
        // In bot mode, join training game
        if (gameSocket && gameSocket.readyState === 1) {
          gameSocket.send('49029#' + JSON.stringify({ mt_id: -1 }));
          log('[BOT] Joining training game...');
        }
      } else {
        var t = pickNextTable();
        if (t) joinTable(t);
        else log('No tables available.');
      }
    });

    document.getElementById('jc-btn-pause').addEventListener('click', function () {
      isPaused = !isPaused;
      this.textContent = isPaused ? 'Resume' : 'Pause';
      this.className = 'jc-btn' + (isPaused ? ' active' : '');
      if (isPaused) {
        clearTimeout(autoJoinTimer);
        clearTimeout(botPlayTimer);
        log('Paused auto-cycling.');
      } else {
        log('Resumed.');
        if (autoJoinEnabled && !isInGame) scheduleAutoJoin(2000);
      }
    });
  }

  function appendLogEntry(entry) {
    var area = document.getElementById('jc-log-area');
    if (!area) return;
    var div = document.createElement('div');
    div.className = 'jc-log-entry' + (entry.isMatch ? ' match' : '');
    div.innerHTML = '<span class="time">' + entry.time + '</span>' +
      entry.text.replace(/</g, '&lt;');
    area.appendChild(div);
    while (area.children.length > MAX_LOG) area.removeChild(area.firstChild);
    area.scrollTop = area.scrollHeight;
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  DASHBOARD — UPDATE CYCLE
  // ═══════════════════════════════════════════════════════════════════════
  function updateDashboard() {
    if (!dashboardReady) return;

    var stats = loadStats();
    var matches = loadMatches();

    // Status dot
    var dot = document.getElementById('jc-status-dot');
    var statusText = document.getElementById('jc-status-text');
    if (dot) dot.className = 'status-dot' + (isInGame ? ' on' : '');
    if (statusText) {
      if (isInGame && botPlayEnabled) statusText.textContent = 'bot playing';
      else if (isInGame) statusText.textContent = 'watching';
      else if (isPaused) statusText.textContent = 'paused';
      else if (autoJoinEnabled) statusText.textContent = 'waiting';
      else statusText.textContent = 'idle';
    }

    // Server status
    var serverEl = document.getElementById('jc-server-status');
    if (serverEl) {
      serverEl.textContent = 'Server: ' + (serverConnected ? 'Connected' : 'Disconnected');
      serverEl.className = 'jc-server-status ' + (serverConnected ? 'connected' : 'disconnected');
    }

    // Mode toggle visual state
    var watchBtn = document.getElementById('jc-mode-watch');
    var assistBtn = document.getElementById('jc-mode-assist');
    var botBtn = document.getElementById('jc-mode-bot');
    if (watchBtn && botBtn && assistBtn) {
      var curMode = botPlayEnabled ? 'bot' : (assistantMode ? 'assist' : 'watch');
      watchBtn.className = 'jc-mode-btn' + (curMode === 'watch' ? ' active-watch' : '');
      assistBtn.className = 'jc-mode-btn' + (curMode === 'assist' ? ' active-bot' : '');
      botBtn.className = 'jc-mode-btn' + (curMode === 'bot' ? ' active-bot' : '');
    }

    // Timing display
    var timingEl = document.getElementById('jc-timing-display');
    if (timingEl && (botPlayEnabled || assistantMode)) {
      var parts = [];
      if (lastBotThinkTime !== null) parts.push('Move: ' + lastBotThinkTime + 'ms');
      if (lastDeclThinkTime !== null) parts.push('Decl: ' + lastDeclThinkTime + 'ms');
      timingEl.textContent = parts.join(' | ');
    } else if (timingEl) {
      timingEl.textContent = '';
    }

    // PNL display
    var pnlEl = document.getElementById('jc-pnl-display');
    if (pnlEl && sessionPNL) {
      var p = sessionPNL;
      var winColor = p.win_rate >= 30 ? '#4CAF50' : (p.win_rate >= 20 ? '#FF9800' : '#e53935');
      var devColor = p.avg_dev <= 8 ? '#4CAF50' : (p.avg_dev <= 12 ? '#FF9800' : '#e53935');
      pnlEl.innerHTML =
        '<span style="color:' + winColor + ';font-weight:bold;font-size:12px;">' +
        p.wins + 'W ' + p.losses + 'L (' + p.win_rate + '%)</span>' +
        ' <span style="color:#555;">|</span> ' +
        '<span style="color:' + devColor + ';">dev ' + p.avg_dev + '</span>' +
        (p.last_10_avg_dev ? ' <span style="color:#888;">(L10: ' + p.last_10_avg_dev + ')</span>' : '') +
        ' <span style="color:#555;">|</span> ' +
        '<span style="color:#6dd5fa;">' + p.perfect_rounds + ' perfect</span>' +
        ' <span style="color:#555;">|</span> ' +
        '<span style="color:#888;">' + p.total_rounds + ' rounds</span>';
    } else if (pnlEl) {
      pnlEl.innerHTML = '<span style="color:#555;">Session: waiting for server...</span>';
    }

    // Stats
    setText('jc-s-matches', matches.length);
    setText('jc-s-rounds', stats.rounds);
    setText('jc-s-complete', stats.fullRounds);

    var elapsed = (Date.now() - stats.started) / 3600000;
    setText('jc-s-rate', elapsed > 0.01 ? (stats.rounds / elapsed).toFixed(1) + '/hr' : '-');

    var totalDevEntries = stats.devHist.length;
    if (totalDevEntries > 0) {
      var avgDev = (stats.devHist.reduce(function (a, b) { return a + b; }, 0) / totalDevEntries).toFixed(1);
      setText('jc-s-avgdev', avgDev);
      var avgEl = document.getElementById('jc-s-avgdev');
      if (avgEl) avgEl.className = 'val ' + (avgDev < 5 ? 'green' : avgDev < 15 ? 'orange' : 'red');
    }

    var perfectPct = stats.rounds > 0
      ? ((stats.perfectRounds || 0) / stats.rounds * 100).toFixed(1) + '%'
      : '-';
    setText('jc-s-perfectpct', perfectPct);

    updateTrumpBars(stats);
    updateDevChart(stats);
    updateRoomDist(stats);
    updateStorageMeter();
    updateCurrentMatch();
    updateLiveGame();
    updateScatter(matches);
    updateHistoryStrip(matches);
    updateLeaderboard();
  }

  function setText(id, val) {
    var el = document.getElementById(id);
    if (el) el.textContent = val;
  }

  // ── Trump bars ──
  function updateTrumpBars(stats) {
    var el = document.getElementById('jc-trump-bars');
    if (!el) return;
    var total = stats.trumpDist.reduce(function (a, b) { return a + b; }, 0) || 1;
    var suits = ['H', 'D', 'S', 'C'];
    el.innerHTML = suits.map(function (s, i) {
      var pct = (stats.trumpDist[i] / total * 100).toFixed(0);
      return '<div class="jc-bar-row">' +
        '<div class="jc-bar-label" style="color:' + SUIT_CLR[s] + '">' + SUIT_SYM[s] + '</div>' +
        '<div class="jc-bar-bg"><div class="jc-bar-fill" style="width:' + pct + '%;background:' + SUIT_CLR[s] + '"></div></div>' +
        '<div class="jc-bar-count">' + stats.trumpDist[i] + '</div></div>';
    }).join('');
  }

  // ── Deviation chart (last 20 rounds, mini bar chart) ──
  function updateDevChart(stats) {
    var el = document.getElementById('jc-dev-chart');
    if (!el) return;
    var hist = stats.devHist.slice(-20);
    if (!hist.length) {
      el.innerHTML = '<div style="color:#555;font-size:9px">No data yet.</div>';
      return;
    }
    var maxD = Math.max.apply(null, hist) || 1;
    el.innerHTML = hist.map(function (d) {
      var h = Math.max(2, (d / maxD) * 38);
      var c = d < 5 ? '#4caf50' : d < 15 ? '#ffa726' : '#ef5350';
      return '<div class="bar" style="height:' + h + 'px;background:' + c + '" title="Avg dev: ' + d + '"></div>';
    }).join('');
  }

  // ── Room distribution ──
  function updateRoomDist(stats) {
    var el = document.getElementById('jc-room-dist');
    if (!el) return;
    var rooms = stats.roomDist || {};
    var entries = Object.entries(rooms).sort(function (a, b) { return b[1] - a[1]; });
    if (!entries.length) {
      el.innerHTML = '<div style="color:#555;font-size:10px">No room data yet.</div>';
      return;
    }
    el.innerHTML = entries.map(function (e) {
      return '<div class="jc-room-row"><span class="name">' + e[0] +
        '</span><span class="count">' + e[1] + '</span></div>';
    }).join('');
  }

  // ── Storage meter ──
  function updateStorageMeter() {
    var used = getStorageUsage();
    var maxBytes = 5 * 1024 * 1024;
    var pct = Math.min(100, used / maxBytes * 100);
    var fillEl = document.getElementById('jc-storage-fill');
    var textEl = document.getElementById('jc-storage-text');
    var warnEl = document.getElementById('jc-storage-warn');
    if (fillEl) {
      fillEl.style.width = pct + '%';
      fillEl.className = 'jc-storage-fill' + (pct > 84 ? ' critical' : pct > 60 ? ' warn' : '');
    }
    if (textEl) textEl.textContent = (used / 1024).toFixed(0) + ' KB / ~5 MB (' + pct.toFixed(0) + '%)';
    if (warnEl) {
      if (used > STORAGE_WARN_BYTES) {
        warnEl.style.display = 'inline';
        warnEl.textContent = used > STORAGE_EXPORT_BYTES ? 'STORAGE CRITICAL' : 'STOR WARN';
        warnEl.style.color = used > STORAGE_EXPORT_BYTES ? '#ef5350' : '#ffa726';
      } else {
        warnEl.style.display = 'none';
      }
    }
  }

  // ── Current match info ──
  function updateCurrentMatch() {
    var el = document.getElementById('jc-match-content');
    if (!el) return;
    if (!currentMatch) {
      el.innerHTML = '<div style="color:#555;font-size:11px">Waiting for game data...</div>';
      return;
    }

    var playerNames = currentMatch.players.map(function (p) { return p.name; }).join(', ');
    var html = '<div class="jc-match-info">' +
      '<div class="item"><div class="lbl">Room</div><div class="val">' + esc(currentMatch.room.title || '-') + '</div></div>' +
      '<div class="item"><div class="lbl">Stake/Prize</div><div class="val">' + currentMatch.room.stake + '/' + currentMatch.room.prize_pool + '</div></div>' +
      '</div>';
    html += '<div class="jc-players-row">Players: ' + esc(playerNames) + '</div>';

    html += '<div class="jc-round-info">';
    if (currentRound) {
      var trumpColor = trumpSuit ? SUIT_CLR[trumpSuit] : '#ccc';
      html += '<div class="jc-trump-display" style="color:' + trumpColor + '">' +
        (SUIT_SYM[trumpSuit] || '?') +
        '<span class="name">' + (SUIT_NAME[trumpSuit] || '?') + '</span></div>' +
        '<div class="item"><div class="lbl">Round</div><div class="val">' + matchRound + '/' + maxRounds + '</div></div>' +
        '<div class="item"><div class="lbl">Trick</div><div class="val">' + (trickNumber + 1) + '/9</div></div>';
    } else {
      html += '<div style="color:#555;font-size:10px">Between rounds...</div>';
    }
    html += '</div>';

    // Deviation bars for completed rounds
    if (currentMatch.rounds.length > 0) {
      html += '<div class="jc-dev-bars">';
      currentMatch.rounds.forEach(function (r, i) {
        var avgDev = 0;
        if (r.results && r.results.results) {
          avgDev = r.results.results.reduce(function (s, res) {
            return s + Math.abs((res.callP || 0) - (res.points || 0));
          }, 0) / 4;
        }
        var maxH = 30;
        var h = Math.max(2, Math.min(maxH, avgDev / 30 * maxH));
        var color = avgDev < 5 ? '#4caf50' : avgDev < 15 ? '#ffa726' : '#ef5350';
        html += '<div class="jc-dev-bar-item">' +
          '<div class="value">' + avgDev.toFixed(0) + '</div>' +
          '<div class="bar" style="height:' + h + 'px;background:' + color + '"></div>' +
          '<div class="label">R' + (i + 1) + '</div></div>';
      });
      html += '</div>';
    }

    el.innerHTML = html;
  }

  // ── Live game display ──
  function updateLiveGame() {
    var el = document.getElementById('jc-live-content');
    if (!el) return;
    if (!currentRound) {
      el.innerHTML = '<div style="color:#555;font-size:11px">No active round.</div>';
      return;
    }

    var html = '';
    html += '<div class="jc-trick-live">';
    for (var seat = 0; seat < 4; seat++) {
      var card = currentTrick ? currentTrick.cards.find(function (c) { return c.player === seat; }) : null;
      var playerName = currentMatch && currentMatch.players[seat] ? currentMatch.players[seat].name : 'Seat ' + seat;
      var shortName = playerName.length > 7 ? playerName.substring(0, 6) + '\u2026' : playerName;
      if (card) {
        html += '<div class="jc-trick-card" style="' + cardColorStyle(card.card) + '">' +
          cardDisplay(card.card) + '<span class="seat">' + esc(shortName) + '</span></div>';
      } else {
        html += '<div class="jc-trick-card empty">__<span class="seat">' + esc(shortName) + '</span></div>';
      }
    }
    html += '</div>';

    // Last completed trick
    if (lastCompletedTrick) {
      var winnerName = currentMatch && currentMatch.players[lastCompletedTrick.winner]
        ? currentMatch.players[lastCompletedTrick.winner].name : 'Seat ' + lastCompletedTrick.winner;
      html += '<div class="jc-last-trick">Last: ' +
        lastCompletedTrick.cards.map(function (c) {
          return '<span style="' + cardColorStyle(c.card) + '">' + cardDisplay(c.card) + '</span>';
        }).join(' ') +
        ' \u2192 <span class="winner">' + esc(winnerName) + '</span> (' + lastCompletedTrick.points + 'pts)</div>';
    }

    // Running scores
    html += '<div class="jc-running-scores">';
    for (var s = 0; s < 4; s++) {
      var name = currentMatch && currentMatch.players[s] ? currentMatch.players[s].name : 'S' + s;
      var sn = name.length > 7 ? name.substring(0, 6) + '\u2026' : name;
      html += '<div class="player-score"><div class="name">' + esc(sn) +
        '</div><div class="pts">' + (roundScores[s] || 0) + '</div></div>';
    }
    html += '</div>';

    el.innerHTML = html;
  }

  // ── Declaration scatter (improved) ──
  function updateScatter(matches) {
    var el = document.getElementById('jc-scatter');
    if (!el) return;

    // Clear previous content entirely
    el.innerHTML = '';

    var maxVal = 157;
    var gridValues = [0, 40, 80, 120, 157];

    // Draw grid lines and labels
    gridValues.forEach(function (v) {
      var pct = v / maxVal * 100;

      // Horizontal grid line
      var lh = document.createElement('div');
      lh.className = 'axis-line-h';
      lh.style.bottom = pct + '%';
      el.appendChild(lh);

      // Vertical grid line
      var lv = document.createElement('div');
      lv.className = 'axis-line-v';
      lv.style.left = pct + '%';
      el.appendChild(lv);

      // Y-axis label (left side)
      var yLabel = document.createElement('div');
      yLabel.className = 'grid-label';
      yLabel.style.left = '2px';
      yLabel.style.bottom = pct + '%';
      yLabel.style.transform = 'translateY(50%)';
      yLabel.textContent = v;
      el.appendChild(yLabel);

      // X-axis label (bottom)
      if (v > 0) {
        var xLabel = document.createElement('div');
        xLabel.className = 'grid-label';
        xLabel.style.bottom = '1px';
        xLabel.style.left = pct + '%';
        xLabel.style.transform = 'translateX(-50%)';
        xLabel.textContent = v;
        el.appendChild(xLabel);
      }
    });

    // Diagonal line (dashed via SVG)
    var svgNs = 'http://www.w3.org/2000/svg';
    var svg = document.createElementNS(svgNs, 'svg');
    svg.setAttribute('class', 'diagonal');
    svg.setAttribute('viewBox', '0 0 100 100');
    svg.setAttribute('preserveAspectRatio', 'none');
    svg.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none';
    var line = document.createElementNS(svgNs, 'line');
    line.setAttribute('x1', '0');
    line.setAttribute('y1', '100');
    line.setAttribute('x2', '100');
    line.setAttribute('y2', '0');
    line.setAttribute('stroke', '#555');
    line.setAttribute('stroke-width', '0.5');
    line.setAttribute('stroke-dasharray', '2,2');
    svg.appendChild(line);
    el.appendChild(svg);

    // Axis labels
    var xAxisLabel = document.createElement('div');
    xAxisLabel.className = 'axis-label';
    xAxisLabel.style.cssText = 'bottom:10px;left:50%;transform:translateX(-50%)';
    xAxisLabel.textContent = 'Declared';
    el.appendChild(xAxisLabel);

    var yAxisLabel = document.createElement('div');
    yAxisLabel.className = 'axis-label';
    yAxisLabel.style.cssText = 'top:50%;left:2px;transform:rotate(-90deg) translateX(-50%);transform-origin:left top';
    yAxisLabel.textContent = 'Scored';
    el.appendChild(yAxisLabel);

    // Count dots
    var dotCount = 0;
    var recent = matches.slice(-50);
    recent.forEach(function (match) {
      if (!match.rounds) return;
      match.rounds.forEach(function (round) {
        if (!round.results || !round.results.results) return;
        round.results.results.forEach(function (res) {
          if (res.callP === undefined || res.points === undefined) return;
          dotCount++;
          var x = res.callP / maxVal * 100;
          var y = res.points / maxVal * 100;
          var dev = Math.abs(res.callP - res.points);
          var color = dev < 5 ? '#4caf50' : dev < 15 ? '#ffa726' : '#ef5350';
          var d = document.createElement('div');
          d.className = 'dot';
          d.style.left = x + '%';
          d.style.bottom = y + '%';
          d.style.background = color;
          d.title = 'Declared: ' + res.callP + ', Scored: ' + res.points + ', Dev: ' + dev;
          el.appendChild(d);
        });
      });
    });

    // Data point count
    var countLabel = document.createElement('div');
    countLabel.className = 'scatter-count';
    countLabel.textContent = dotCount + ' declarations';
    el.appendChild(countLabel);
  }

  // ── Match history strip ──
  function updateHistoryStrip(matches) {
    var el = document.getElementById('jc-history-strip');
    if (!el) return;
    var recent = matches.slice(-20);
    if (!recent.length) {
      el.innerHTML = '<div style="color:#555;font-size:10px">No matches yet.</div>';
      return;
    }
    var offset = matches.length - recent.length;
    el.innerHTML = recent.map(function (m, idx) {
      var avgDev = 0, count = 0;
      if (m.rounds) {
        m.rounds.forEach(function (r) {
          if (r.results && r.results.results) {
            r.results.results.forEach(function (res) {
              avgDev += Math.abs((res.callP || 0) - (res.points || 0));
              count++;
            });
          }
        });
      }
      avgDev = count > 0 ? avgDev / count : 0;
      var color = avgDev < 5 ? '#4caf50' : avgDev < 12 ? '#ffa726' : '#ef5350';
      var rounds = m.rounds ? m.rounds.length : 0;
      return '<div class="jc-history-block" style="background:' + color +
        '" title="Match ' + (offset + idx + 1) + ': ' + rounds + 'R, avg dev ' +
        avgDev.toFixed(1) + '">' + rounds + '</div>';
    }).join('');
  }

  // ── Player leaderboard ──
  function updateLeaderboard() {
    var body = document.getElementById('jc-lb-body');
    if (!body) return;

    updatePlayerStats();

    var players = Object.values(allPlayers)
      .filter(function (p) { return p.rounds >= 2; })
      .sort(function (a, b) { return (a.totalDev / a.rounds) - (b.totalDev / b.rounds); });

    if (!players.length) {
      body.innerHTML = '<tr><td colspan="7" style="color:#555;text-align:center;padding:8px">' +
        'No player data yet (need 2+ rounds).</td></tr>';
      return;
    }

    body.innerHTML = players.slice(0, 50).map(function (p, i) {
      var avgDev = (p.totalDev / p.rounds).toFixed(1);
      var devClass = avgDev < 5 ? 'green' : avgDev < 12 ? 'orange' : 'red';

      var sparkline = '';
      if (p.deviations.length > 1) {
        var maxD = Math.max.apply(null, p.deviations) || 1;
        sparkline = '<span class="jc-sparkline">' +
          p.deviations.slice(-8).map(function (d) {
            var h = Math.max(1, (d / maxD) * 10);
            var c = d < 5 ? '#4caf50' : d < 12 ? '#ffa726' : '#ef5350';
            return '<span class="bar" style="height:' + h + 'px;background:' + c + '"></span>';
          }).join('') + '</span>';
      }

      return '<tr>' +
        '<td class="rank">' + (i + 1) + '</td>' +
        '<td class="player-name">' + esc(p.name) + '</td>' +
        '<td class="num">' + p.rounds + '</td>' +
        '<td class="num ' + devClass + '">' + avgDev + '</td>' +
        '<td class="num green">' + (p.bestDev === 999 ? '-' : p.bestDev) + '</td>' +
        '<td class="num green">' + p.perfectCount + '</td>' +
        '<td>' + sparkline + '</td></tr>';
    }).join('');
  }

  // ── Utility: escape HTML ──
  function esc(str) {
    if (!str) return '';
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  INIT
  // ═══════════════════════════════════════════════════════════════════════
  function init() {
    log('Jass Collector v7.0 | Tab ' + tabNumber);
    createDashboard();
    updateTitle();

    // Periodic refresh
    setInterval(updateDashboard, 2000);

    // No auto-dismiss — the Ansage UI overlay is cosmetic only.
    // The declaration goes through via WebSocket. The overlay doesn't affect gameplay.

    // Server status check
    checkServerStatus();
    setInterval(checkServerStatus, SERVER_CHECK_INTERVAL);

    // Save on unload
    window.addEventListener('beforeunload', function () {
      if (currentRound && currentRound.tricks.length > 0) {
        currentRound.results = currentRound.results || null;
        finishRound();
      }
      if (currentMatch && currentMatch.rounds.length > 0) finalizeMatch();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
