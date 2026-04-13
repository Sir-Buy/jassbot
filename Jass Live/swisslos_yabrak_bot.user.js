// ==UserScript==
// @name         Swisslos Yabrak Bot
// @namespace    yabrak-bot
// @version      2.0.0
// @description  Differenzler assistant with V7 RL engine — shows best card, tracks PNL
// @match        https://www.swisslos.ch/de/jass/differenzler/spielen.html
// @match        https://www.swisslos.ch/fr/jass/differenzler/jouer.html
// @match        https://www.swisslos.ch/it/jass/differenzler/giocare.html
// @match        https://www.swisslos.ch/en/jass/differenzler/play.html
// @grant        none
// @run-at       document-start
// ==/UserScript==

(function () {
  'use strict';

  const BOT_URL = 'http://localhost:5000';
  const MY_NAME = 'abc3';  // your Swisslos player name
  const AUTO_PLAY = false;  // disabled — manual play only

  // ── Card System ──
  const SUIT_ID  = { H: 0, D: 1, S: 2, C: 3 };
  const SUIT_SYM = { H: '\u2665', D: '\u2666', S: '\u2660', C: '\u2663' };
  const SUIT_CLR = { H: '#e53935', D: '#FF9800', S: '#64b5f6', C: '#aaa' };
  const SUIT_BG  = { H: '#2a1010', D: '#2a1a08', S: '#0a1a2a', C: '#1a1a1a' };
  const VAL_ID   = { '6': 0, '7': 1, '8': 2, '9': 3, '10': 4, J: 5, Q: 6, K: 7, A: 8 };
  const BASE_PTS  = { '6': 0, '7': 0, '8': 0, '9': 0, '10': 10, J: 2, Q: 3, K: 4, A: 11 };
  const TRUMP_PTS = { '6': 0, '7': 0, '8': 0, '9': 14, '10': 10, J: 20, Q: 3, K: 4, A: 11 };

  function cardPoints(code, trump) {
    if (!code) return 0;
    var s = code.charAt(0), v = code.substring(1);
    return (s === trump ? TRUMP_PTS[v] : BASE_PTS[v]) || 0;
  }

  function cardDisplay(code) {
    if (!code) return '?';
    return (SUIT_SYM[code[0]] || code[0]) + code.substring(1);
  }

  function cardSpan(code, highlight) {
    var c = SUIT_CLR[code[0]] || '#aaa';
    var bg = highlight ? '#2e7d32' : (SUIT_BG[code[0]] || '#111');
    var border = highlight ? '2px solid #4CAF50' : '1px solid #333';
    var size = highlight ? 'font-size:22px' : 'font-size:16px';
    return '<span style="color:' + c + ';background:' + bg + ';border:' + border +
      ';border-radius:6px;padding:3px 7px;font-weight:bold;' + size + '">' +
      cardDisplay(code) + '</span>';
  }

  // ── State ──
  var gameSocket = null;
  var myPos = -1;
  var myHand = [];
  var trumpSuit = '';
  var seqNr = 0;
  var trickNumber = 0;
  var currentTrick = null;
  var cardsInTrick = 0;
  var roundScores = [0, 0, 0, 0];
  var serverOk = false;
  var engineName = '';
  var pnl = null;
  var matchPlayers = [];
  var matchId = '';
  var roundHistory = [];
  var allRounds = [];
  var initialHand = [];
  var lastTrickProcessed = -1;  // dedup trick processing
  var matchCumDevs = {};  // {pos: cumulative deviation} from results
  var roundResults = [];  // [{trump, decl, scored, dev, rank}] this match
  var knownVoids = {};    // {pos: ['H','S',...]} from server

  // ── Server ──
  function botFetch(path, body) {
    return fetch(BOT_URL + path, {
      method: body ? 'POST' : 'GET',
      headers: body ? { 'Content-Type': 'application/json' } : {},
      body: body ? JSON.stringify(body) : undefined,
    }).then(function (r) { return r.json(); });
  }

  function checkServer() {
    botFetch('/status')
      .then(function (r) {
        serverOk = true;
        engineName = r.engine || '?';
        if (r.session) pnl = r.session;
        updateUI();
      })
      .catch(function () { serverOk = false; updateUI(); });
  }
  setInterval(checkServer, 5000);
  setTimeout(checkServer, 500);

  // ── WebSocket Intercept ──
  var OrigWS = window.WebSocket;
  window.WebSocket = function (url, protocols) {
    var ws = protocols ? new OrigWS(url, protocols) : new OrigWS(url);
    if (url && url.includes('swisslos')) {
      gameSocket = ws;
      ws.addEventListener('message', function (evt) {
        try { parseMsg(evt.data); } catch (e) { console.error('[YB]', e); }
      });
    }
    return ws;
  };
  window.WebSocket.prototype = OrigWS.prototype;
  Object.assign(window.WebSocket, OrigWS);

  function parseMsg(raw) {
    if (typeof raw !== 'string') return;
    if (raw === '2') { gameSocket.send('3'); return; }
    if (!raw.startsWith('4')) return;
    var rest = raw.substring(1);
    var hashIdx = rest.indexOf('#');
    if (hashIdx < 0) return;
    var msgId = rest.substring(0, hashIdx);
    var body;
    try { body = JSON.parse(rest.substring(hashIdx + 1)); } catch (e) { return; }
    handleMsg(msgId, body);
  }

  // ── Protocol Intelligence ──
  var seenMsgIds = {};
  var declPhase = false;  // true between 9022 and first msg 5
  var interceptedDeclarations = {};  // {swisslos_pos: declared_points}
  var msgLog = [];  // log all messages during declaration phase

  function handleMsg(id, body) {
    // Track all unique message IDs we see
    if (!seenMsgIds[id]) {
      seenMsgIds[id] = true;
      console.log('[YB] NEW MSG TYPE:', id, JSON.stringify(body).substring(0, 500));
    }

    // During declaration phase, log EVERYTHING — looking for leaked declarations
    if (declPhase && id !== '9022') {
      var entry = { id: id, ts: Date.now(), body: body };
      msgLog.push(entry);
      console.log('[YB-DECL-PHASE] msg ' + id + ':', JSON.stringify(body).substring(0, 300));

      // Look for any field that could be a declaration value
      if (body && typeof body === 'object') {
        // Check for callP, points, declaration, decl, ansage, predict fields
        var suspicious = ['callP', 'points', 'declaration', 'decl', 'ansage',
                          'predict', 'call', 'bid', 'estimate', 'target'];
        suspicious.forEach(function (key) {
          if (body[key] !== undefined) {
            console.warn('[YB-INTEL] Found "' + key + '" = ' + body[key] + ' in msg ' + id);
          }
        });
        // Check for player position + numeric value combos
        if (body.pos !== undefined && body.points !== undefined) {
          console.warn('[YB-INTEL] pos=' + body.pos + ' points=' + body.points + ' in msg ' + id);
        }
        // Check for arrays of results
        if (body.results && Array.isArray(body.results)) {
          body.results.forEach(function (r) {
            if (r.callP !== undefined) {
              console.warn('[YB-INTEL] DECLARATION FOUND: pos=' + r.pos + ' callP=' + r.callP);
              interceptedDeclarations[r.pos] = r.callP;
            }
          });
        }
      }
    }

    switch (id) {
      case '9022': onGameInit(body); break;
      case '1': onCardsDealt(body); break;
      case '2': onDeclStatus(body); break;   // might be declaration status
      case '3': onDeclStatus(body); break;   // might be declaration confirmation
      case '4': onDeclStatus(body); break;   // might be game state update
      case '5': onPlayable(body); break;
      case '6': onCardPlayed(body); break;
      case '7': onCardPlayed(body); break;
      case '8': onTrickWon(body); break;
      case '9': onRemoveTrick(body); break;
      case '10': case '17': onResult(body); break;
      case '19': onTrump(body); break;
      default:
        // Log any unknown message during gameplay
        if (trickNumber >= 0 && parseInt(id) < 100) {
          console.log('[YB-UNKNOWN] msg ' + id + ':', JSON.stringify(body).substring(0, 300));
        }
        break;
    }
  }

  function onDeclStatus(body) {
    // Messages 2/3/4 during declaration phase — check for leaked info
    if (!body || typeof body !== 'object') return;
    // If body has a pos and any numeric value, it might be a declaration
    if (body.pos !== undefined) {
      console.log('[YB-DECL-STATUS] msg with pos=' + body.pos, JSON.stringify(body));
      if (body.points !== undefined && declPhase) {
        interceptedDeclarations[body.pos] = body.points;
        console.warn('[YB-INTEL] Possible declaration: pos=' + body.pos + ' val=' + body.points);
      }
      if (body.callP !== undefined) {
        interceptedDeclarations[body.pos] = body.callP;
        console.warn('[YB-INTEL] Declaration leaked: pos=' + body.pos + ' callP=' + body.callP);
      }
    }
  }

  // ── Game Events ──
  function onGameInit(d) {
    if (!d) return;
    trickNumber = 0;
    currentTrick = null;
    cardsInTrick = 0;
    roundScores = [0, 0, 0, 0];
    roundHistory = [];
    lastTrickProcessed = -1;
    declPhase = true;  // start sniffing for declarations
    interceptedDeclarations = {};
    msgLog = [];
    console.log('[YB] === ROUND INIT === Full 9022 body:', JSON.stringify(d).substring(0, 1000));

    if (d.players && Array.isArray(d.players)) {
      matchPlayers = d.players;
    }

    // Seat detection — three methods, in priority order
    myPos = -1;
    // 1. Server-provided pos field (training games)
    if (d.pos !== undefined && d.pos >= 0) {
      myPos = d.pos;
      console.log('[YB] Seat from data.pos:', myPos);
    }
    // 2. Match by configured player name
    if (myPos < 0 && matchPlayers.length > 0) {
      for (var i = 0; i < matchPlayers.length; i++) {
        var p = matchPlayers[i];
        if (p && p.name && p.name.toLowerCase() === MY_NAME.toLowerCase()) {
          myPos = p.pos !== undefined ? p.pos : i;
          console.log('[YB] Seat from name match:', myPos, p.name);
          break;
        }
      }
    }
    // 3. Guest player (uid === null or name starts with Gast)
    if (myPos < 0 && matchPlayers.length > 0) {
      for (var i = 0; i < matchPlayers.length; i++) {
        var p = matchPlayers[i];
        if (p && (p.uid === null || (p.name && p.name.indexOf('Gast') >= 0))) {
          myPos = p.pos !== undefined ? p.pos : i;
          console.log('[YB] Seat from guest fallback:', myPos);
          break;
        }
      }
    }
    // Warn if detection failed
    if (myPos < 0) {
      var names = matchPlayers.map(function (p) { return (p.name || '?') + '(uid=' + p.uid + ')'; });
      console.error('[YB] SEAT DETECTION FAILED! Players:', names.join(', '), '| MY_NAME:', MY_NAME);
      showRec('SEAT DETECTION FAILED<br>Set MY_NAME in script to your player name<br>Players: ' +
        names.join(', '), '#e53935');
    }

    if (d.t_id) {
      var newMatch = String(d.t_id);
      if (newMatch !== matchId) {
        // New match — reset match-level state
        matchCumDevs = {};
        roundResults = [];
        allRounds = [];
      }
      matchId = newMatch;
    }
    if (d.trump && SUIT_ID[d.trump] !== undefined) trumpSuit = d.trump;
    knownVoids = {};

    // Sync seqNr from server
    if (d.seq !== undefined) seqNr = d.seq;

    if (d.cards && Array.isArray(d.cards)) {
      myHand = d.cards.slice();
      initialHand = d.cards.slice();
      if (d.selectScreen && serverOk) {
        botFetch('/new-round', { hand: myHand, trump: trumpSuit, my_pos: myPos })
          .then(function (resp) {
            var decl = resp.declaration !== undefined ? resp.declaration : '?';
            showRec('DECLARE: <b style="font-size:28px">' + decl + '</b> points', '#2196F3');
          })
          .catch(function (e) { showRec('Server error: ' + e.message, '#e53935'); });
      }
    }
    updateUI();
  }

  function onCardsDealt(d) {
    if (d && d.cards) myHand = d.cards.slice();
  }

  function onTrump(d) {
    if (d) {
      var t = d.trump || d.t;
      if (t && SUIT_ID[t] !== undefined) trumpSuit = t;
    }
  }

  function onPlayable(body) {
    if (!body || !body.pCards || !body.pCards.length) return;
    var pCards = body.pCards;

    // End declaration sniffing phase
    if (declPhase) {
      declPhase = false;
      console.log('[YB] Declaration phase ended. Messages captured:', msgLog.length);
      if (msgLog.length > 0) {
        console.log('[YB] Decl phase messages:', JSON.stringify(msgLog));
      }
      // Send any intercepted declarations to server
      if (Object.keys(interceptedDeclarations).length > 0 && serverOk) {
        console.warn('[YB-INTEL] Sending intercepted declarations:', interceptedDeclarations);
        botFetch('/opponent-info', { declarations: interceptedDeclarations }).catch(function () {});
      }
    }

    if (!serverOk) { showRec('Server offline', '#e53935'); return; }

    var trickCards = [];
    if (currentTrick && currentTrick.cards) {
      currentTrick.cards.forEach(function (c) {
        trickCards.push({ player: c.player, card: c.card });
      });
    }
    var leader = currentTrick ? currentTrick.leader : 0;
    var oppPoints = [];
    for (var s = 0; s < 4; s++) {
      if (s !== myPos) oppPoints.push(roundScores[s] || 0);
    }

    botFetch('/play', {
      hand: myHand, trick_cards: trickCards,
      leader: leader, legal_moves: pCards, opp_points: oppPoints,
    }).then(function (resp) {
      var card = resp.card_code || pCards[0];
      var gap = resp.gap;
      var target = resp.target;
      // Use our own tracked points (more reliable than server's)
      var scored = myPos >= 0 ? (roundScores[myPos] || 0) : (resp.my_points || 0);
      // Capture voids from server
      if (resp.voids) knownVoids = resp.voids;

      // Build hand display
      var handHtml = '<div style="display:flex;gap:4px;flex-wrap:wrap;margin:8px 0">';
      var isLegal = {};
      pCards.forEach(function (c) { isLegal[c] = true; });
      myHand.forEach(function (c) {
        var isRec = c === card;
        var playable = isLegal[c];
        if (isRec) {
          handHtml += cardSpan(c, true);
        } else if (playable) {
          handHtml += '<span style="opacity:0.7">' + cardSpan(c, false) + '</span>';
        } else {
          handHtml += '<span style="opacity:0.3">' + cardSpan(c, false) + '</span>';
        }
      });
      handHtml += '</div>';

      // Gap info (recalculate from our tracked points)
      var myGap = target - scored;
      var gapHtml = '';
      if (myGap > 0) gapHtml = '<span style="color:#FF9800">Need ' + myGap + ' more</span>';
      else if (myGap === 0) gapHtml = '<span style="color:#4CAF50">On target!</span>';
      else gapHtml = '<span style="color:#e53935">' + Math.abs(myGap) + ' over</span>';

      var autoTag = AUTO_PLAY ? ' <span style="color:#6dd5fa;font-size:10px">[AUTO]</span>' : '';
      showRec(
        'PLAY: ' + cardSpan(card, true) + autoTag +
        '<span style="color:#888;font-size:12px;margin-left:8px">' +
        (resp.time_ms || '') + 'ms</span>' +
        handHtml +
        '<div style="font-size:12px;color:#aaa">' +
        'Target: ' + (target || '?') + ' | Scored: ' + scored +
        ' | ' + gapHtml +
        ' | Trick ' + (trickNumber + 1) + '/9</div>',
        '#4CAF50'
      );

      // Store target for progress bar
      var recEl = document.getElementById('yb-rec');
      if (recEl) recEl.dataset.target = target;

      // Auto-play: click the card on the Swisslos UI
      if (AUTO_PLAY && card) {
        var delay = AUTO_PLAY_DELAY_MIN + Math.random() * (AUTO_PLAY_DELAY_MAX - AUTO_PLAY_DELAY_MIN);
        setTimeout(function () { autoClickCard(card); }, delay);
      }
    }).catch(function (e) {
      showRec('Play error: ' + e.message, '#e53935');
    });
  }

  function onCardPlayed(d) {
    if (!d) return;
    var pos = d.pos !== undefined ? d.pos : cardsInTrick;
    var code = d.card || d.c;
    if (!code) return;
    if (!currentTrick) { currentTrick = { leader: pos, cards: [] }; cardsInTrick = 0; }
    // Dedup: don't add same card+player twice (msg 6 and 7 can overlap)
    if (currentTrick.cards.some(function (c) { return c.card === code && c.player === pos; })) return;
    var pts = cardPoints(code, trumpSuit);
    currentTrick.cards.push({ player: pos, card: code, points: pts });
    cardsInTrick++;
    // Remove from our hand
    var idx = myHand.indexOf(code);
    if (idx >= 0) myHand.splice(idx, 1);
  }

  function onTrickWon(d) {
    if (!d) return;
    // Dedup: don't process same trick twice
    if (trickNumber === lastTrickProcessed) return;
    lastTrickProcessed = trickNumber;

    var winner = d.pos;

    // Calculate trick points from cards (reliable, not dependent on server msg format)
    var trickPts = 0;
    if (currentTrick && currentTrick.cards.length > 0) {
      trickPts = currentTrick.cards.reduce(function (sum, c) { return sum + (c.points || 0); }, 0);
    }
    // Last trick bonus: +5 points for winning trick 9
    if (trickNumber >= 8) trickPts += 5;

    if (winner >= 0 && winner < 4) roundScores[winner] += trickPts;

    if (currentTrick && currentTrick.cards.length > 0) {
      var tc = currentTrick.cards.map(function (c) { return { player: c.player, card: c.card }; });
      roundHistory.push({ trick_number: trickNumber, cards: tc, winner: winner, points: trickPts });
      if (serverOk) {
        botFetch('/notify-trick', { trick: tc, winner: winner, points: trickPts }).catch(function () {});
      }
    }
    trickNumber++;
    currentTrick = null;
    cardsInTrick = 0;
    updateUI();
  }

  function onRemoveTrick(d) {
    // Server requests trick cleanup — force-complete if we have a full trick
    if (currentTrick && currentTrick.cards.length >= 4) {
      onTrickWon(d || {});
    }
  }

  function onResult(d) {
    if (!d || !d.results) return;
    var results = d.results;
    var myRes = null;
    var myRank = 0;
    results.forEach(function (r) { if (r.pos === myPos) { myRes = r; myRank = r.rank || 0; } });

    if (myRes) {
      var dev = Math.abs((myRes.callP || 0) - (myRes.points || 0));
      var color = dev === 0 ? '#4CAF50' : (dev <= 5 ? '#FF9800' : '#e53935');
      var rankStr = myRank === 1 ? ' | RANK: 1st!' : (myRank ? ' | Rank: ' + myRank : '');
      showRec(
        'Declared: ' + myRes.callP + ' | Scored: ' + myRes.points +
        ' | <b>Dev: ' + dev + '</b>' + rankStr,
        color
      );
    }

    // Track match standings and log declarations
    results.forEach(function (r) {
      matchCumDevs[r.pos] = r.total || 0;
      console.log('[YB]   pos=' + r.pos + ' declared=' + r.callP + ' scored=' + r.points +
        ' dev=' + Math.abs((r.callP||0) - (r.points||0)) + ' rank=' + r.rank +
        ' cumDev=' + r.total);
    });
    if (myRes) {
      roundResults.push({
        trump: trumpSuit,
        decl: myRes.callP || 0,
        scored: myRes.points || 0,
        dev: Math.abs((myRes.callP || 0) - (myRes.points || 0)),
        rank: myRes.rank || 0,
      });
    }

    // Save round data and send end-round (only when seat is detected)
    if (serverOk && myPos >= 0) {
      var opponents = [];
      matchPlayers.forEach(function (p) {
        if (p && p.pos !== myPos) opponents.push(p.name || ('P' + p.pos));
      });

      // Build full round record for analysis
      // Store results as the full msg body (dict with .results key) so server can parse
      var roundData = {
        timestamp: new Date().toISOString(),
        match_id: matchId,
        trump: trumpSuit,
        my_pos: myPos,
        my_hand: initialHand.slice(),
        tricks: roundHistory,
        results: d,  // full msg body: {results: [{pos, callP, points, ...}]}
        scores: roundScores.slice(),
        players: matchPlayers.map(function (p) {
          return { name: p.name || '?', uid: p.uid, pos: p.pos };
        }),
      };
      allRounds.push(roundData);

      // Save game to disk
      botFetch('/save', {
        match_id: matchId,
        players: matchPlayers,
        rounds: allRounds,
      }).catch(function (e) { console.error('[YB] Save error:', e); });

      // Send end-round for PNL
      botFetch('/end-round', { rank: myRank, opponents: opponents, match_id: matchId })
        .then(function (resp) {
          if (resp && resp.session) { pnl = resp.session; updateUI(); }
        })
        .catch(function () {});
    }
  }

  // ── Auto-Play ──
  function autoClickCard(cardCode) {
    // Swisslos renders playable cards as clickable elements.
    // Card images/elements have data attributes or alt text with the card code.
    // Try multiple selectors to find the right card.
    var clicked = false;

    // Method 1: Find by data-card attribute
    var els = document.querySelectorAll('[data-card="' + cardCode + '"]');
    if (els.length > 0) { els[0].click(); clicked = true; }

    // Method 2: Find by card image alt/title containing the code
    if (!clicked) {
      var imgs = document.querySelectorAll('img[alt*="' + cardCode + '"], img[title*="' + cardCode + '"]');
      if (imgs.length > 0) { imgs[0].click(); clicked = true; }
    }

    // Method 3: Find clickable card elements by class and card code in src/alt
    if (!clicked) {
      var allEls = document.querySelectorAll('.card, .playable, [class*="card"]');
      for (var i = 0; i < allEls.length; i++) {
        var el = allEls[i];
        var src = (el.src || el.style.backgroundImage || '').toUpperCase();
        var alt = (el.alt || el.title || el.getAttribute('data-value') || '').toUpperCase();
        var code = cardCode.toUpperCase();
        if (src.indexOf(code) >= 0 || alt.indexOf(code) >= 0) {
          el.click(); clicked = true; break;
        }
      }
    }

    // Method 4: Send the card via WebSocket directly (most reliable)
    if (!clicked && gameSocket && gameSocket.readyState === 1) {
      seqNr++;
      var msg = '46#' + JSON.stringify({ card: cardCode, seqNr: seqNr });
      gameSocket.send(msg);
      clicked = true;
      console.log('[YB-AUTO] Sent card via WebSocket:', cardCode, 'seqNr:', seqNr);
    }

    if (clicked) {
      console.log('[YB-AUTO] Played:', cardCode);
    } else {
      console.warn('[YB-AUTO] Could not auto-play:', cardCode);
    }
  }

  // ── Recommendation Display ──
  function showRec(html, color) {
    var el = document.getElementById('yb-rec');
    if (el) {
      el.innerHTML = html;
      el.style.borderLeftColor = color;
    }
  }

  // ── UI ──
  function createUI() {
    if (document.getElementById('yb-panel')) return;

    var panel = document.createElement('div');
    panel.id = 'yb-panel';
    panel.innerHTML =
      '<div id="yb-header" style="font-size:16px;font-weight:bold;margin-bottom:6px;' +
        'padding-bottom:6px;border-bottom:1px solid #333;display:flex;justify-content:space-between;align-items:center">' +
        '<span>YABRAK BOT ' + (AUTO_PLAY ? '<span style="color:#4CAF50;font-size:10px">AUTO</span>' : '') + '</span>' +
        '<span id="yb-engine" style="font-size:11px;color:#6dd5fa;font-weight:normal"></span>' +
      '</div>' +
      '<div id="yb-pnl" style="padding:6px 8px;margin-bottom:8px;background:#0a0e14;' +
        'border-radius:6px;font-size:13px;font-family:monospace"></div>' +
      '<div id="yb-status" style="font-size:12px;margin-bottom:6px"></div>' +
      '<div id="yb-progress" style="margin:6px 0"></div>' +
      '<div id="yb-rec" style="padding:12px;margin:8px 0;background:#111;border-radius:8px;' +
        'border-left:4px solid #666;min-height:40px;font-size:16px;line-height:1.5"></div>' +
      '<div id="yb-scores" style="font-size:11px;color:#ccc;margin:6px 0;padding:6px 8px;' +
        'background:#0a0e14;border-radius:6px;font-family:monospace"></div>' +
      '<div id="yb-match" style="font-size:11px;color:#ccc;margin:4px 0;padding:6px 8px;' +
        'background:#0a0e14;border-radius:6px;font-family:monospace"></div>' +
      '<div id="yb-info" style="font-size:10px;color:#555;margin-top:4px"></div>';

    panel.style.cssText =
      'position:fixed;top:10px;left:10px;width:360px;max-height:90vh;overflow-y:auto;z-index:99999;' +
      'background:#12121f;color:#eee;border-radius:12px;padding:14px;' +
      'font-family:system-ui,sans-serif;box-shadow:0 4px 24px rgba(0,0,0,0.6);' +
      'cursor:move;user-select:none;border:1px solid #2a2a3a';

    // Drag
    panel.onmousedown = function (e) {
      if (e.target.tagName === 'BUTTON' || e.target.tagName === 'A') return;
      var mx = e.clientX, my = e.clientY;
      document.onmousemove = function (ev) {
        panel.style.top = (panel.offsetTop - (my - ev.clientY)) + 'px';
        panel.style.left = (panel.offsetLeft - (mx - ev.clientX)) + 'px';
        panel.style.right = 'auto';
        mx = ev.clientX; my = ev.clientY;
      };
      document.onmouseup = function () { document.onmousemove = null; };
    };

    document.body.appendChild(panel);
  }

  function updateUI() {
    if (!document.getElementById('yb-panel')) createUI();

    // Engine
    var engEl = document.getElementById('yb-engine');
    if (engEl) engEl.textContent = serverOk ? engineName : 'OFFLINE';

    // Status
    var statusEl = document.getElementById('yb-status');
    if (statusEl) {
      var dot = serverOk ? '\uD83D\uDFE2' : '\uD83D\uDD34';
      var trumpHtml = trumpSuit ? ' | Trump: <span style="color:' + (SUIT_CLR[trumpSuit] || '#aaa') +
        ';font-weight:bold">' + (SUIT_SYM[trumpSuit] || '-') + '</span>' : '';
      statusEl.innerHTML = dot + ' ' + (serverOk ? 'Connected' : 'Offline') +
        ' | Seat: ' + (myPos >= 0 ? myPos : '?') + trumpHtml +
        ' | Trick ' + (trickNumber + 1) + '/9';
    }

    // PNL
    var pnlEl = document.getElementById('yb-pnl');
    if (pnlEl) {
      if (pnl && pnl.total_rounds > 0) {
        var wr = pnl.win_rate;
        var wrColor = wr >= 30 ? '#4CAF50' : (wr >= 20 ? '#FF9800' : '#e53935');
        var devColor = pnl.avg_dev <= 5 ? '#4CAF50' : (pnl.avg_dev <= 10 ? '#FF9800' : '#e53935');
        // Calculate streak
        var games = pnl.games || [];
        var streak = 0;
        var streakType = '';
        for (var gi = games.length - 1; gi >= 0; gi--) {
          var isWin = games[gi].rank === 1;
          if (gi === games.length - 1) { streakType = isWin ? 'W' : 'L'; streak = 1; }
          else if ((isWin && streakType === 'W') || (!isWin && streakType === 'L')) streak++;
          else break;
        }
        var streakColor = streakType === 'W' ? '#4CAF50' : '#e53935';
        var streakHtml = streak > 1 ? '  <span style="color:' + streakColor + '">' + streak + streakType + ' streak</span>' : '';
        // Perfect rate
        var perfectPct = (pnl.perfect_rounds / pnl.total_rounds * 100).toFixed(0);
        pnlEl.innerHTML =
          '<div style="margin-bottom:4px">' +
          '<span style="color:' + wrColor + ';font-weight:bold;font-size:18px">' +
          pnl.wins + 'W ' + pnl.losses + 'L</span>' +
          '  <span style="color:' + wrColor + ';font-size:14px">' + wr + '%</span>' +
          streakHtml + '</div>' +
          '<div>' +
          '<span style="color:' + devColor + '">avg dev ' + pnl.avg_dev + '</span>' +
          (pnl.last_10_avg_dev ? '  <span style="color:#888">(L10: ' + pnl.last_10_avg_dev + ')</span>' : '') +
          '  <span style="color:#555">|</span>  ' +
          '<span style="color:#6dd5fa">' + pnl.perfect_rounds + ' perfect (' + perfectPct + '%)</span>' +
          '  <span style="color:#555">|</span>  ' +
          '<span style="color:#888">' + pnl.total_rounds + ' rds</span></div>';
      } else {
        pnlEl.innerHTML = '<span style="color:#555">No rounds played yet</span>';
      }
    }

    // Progress bar — target vs scored
    var progEl = document.getElementById('yb-progress');
    if (progEl && serverOk) {
      var myScored = myPos >= 0 ? (roundScores[myPos] || 0) : 0;
      var myTarget = 0;
      // Get target from PNL or last known
      if (pnl && pnl.games && pnl.games.length > 0) {
        myTarget = pnl.games[pnl.games.length - 1].target || 0;
      }
      // Fallback: check if we have a target from the play response
      var recEl = document.getElementById('yb-rec');
      if (recEl && recEl.dataset && recEl.dataset.target) myTarget = parseInt(recEl.dataset.target) || 0;

      if (myTarget > 0 || myScored > 0) {
        var maxVal = Math.max(myTarget, myScored, 80);
        var tPct = Math.min(100, (myTarget / maxVal) * 100);
        var sPct = Math.min(100, (myScored / maxVal) * 100);
        var barColor = Math.abs(myTarget - myScored) <= 5 ? '#4CAF50' : (myScored > myTarget ? '#e53935' : '#FF9800');
        progEl.innerHTML =
          '<div style="position:relative;height:20px;background:#1a1a2a;border-radius:4px;overflow:hidden">' +
          '<div style="position:absolute;height:100%;width:' + sPct + '%;background:' + barColor + ';opacity:0.7;border-radius:4px"></div>' +
          '<div style="position:absolute;left:' + tPct + '%;top:0;bottom:0;width:2px;background:#fff"></div>' +
          '<div style="position:relative;padding:2px 6px;font-size:10px;color:#fff;font-family:monospace">' +
          'Scored: ' + myScored + ' / Target: ' + myTarget + '</div></div>';
      } else {
        progEl.innerHTML = '';
      }
    }

    // Scores — player scores with void indicators
    var scoresEl = document.getElementById('yb-scores');
    if (scoresEl) {
      var html = '';
      for (var si = 0; si < 4; si++) {
        var pName = matchPlayers[si] ? (matchPlayers[si].name || 'P' + si).substring(0, 10) : 'P' + si;
        var isMe = si === myPos;
        var pts = roundScores[si] || 0;
        var color = isMe ? '#6dd5fa' : '#aaa';
        var marker = isMe ? ' *' : '';
        // Void indicators
        var voidStr = '';
        if (!isMe && knownVoids[si]) {
          voidStr = ' <span style="color:#555">[void: ' + knownVoids[si].join('') + ']</span>';
        }
        // Score bar
        var barW = Math.min(100, (pts / 80) * 100);
        var barC = isMe ? '#2a4a6a' : '#1a1a2a';
        html += '<div style="margin:1px 0">' +
          '<span style="color:' + color + ';display:inline-block;width:90px">' + pName + marker + '</span>' +
          '<span style="color:' + color + ';display:inline-block;width:30px;text-align:right">' + pts + '</span>' +
          voidStr +
          '<div style="height:3px;background:#0a0a14;border-radius:2px;margin-top:1px">' +
          '<div style="height:100%;width:' + barW + '%;background:' + color + ';border-radius:2px;opacity:0.4"></div></div></div>';
      }
      scoresEl.innerHTML = html;
    }

    // Match standings — cumulative deviations + round history
    var matchEl = document.getElementById('yb-match');
    if (matchEl) {
      var mHtml = '';
      // Round history
      if (roundResults.length > 0) {
        mHtml += '<div style="margin-bottom:4px">';
        roundResults.forEach(function (r, i) {
          var devColor = r.dev === 0 ? '#4CAF50' : (r.dev <= 5 ? '#FF9800' : '#e53935');
          var rankColor = r.rank === 1 ? '#4CAF50' : '#888';
          var trumpClr = SUIT_CLR[r.trump] || '#aaa';
          mHtml += '<span style="margin-right:8px">' +
            '<span style="color:' + trumpClr + '">' + (SUIT_SYM[r.trump] || '?') + '</span> ' +
            '<span style="color:' + devColor + '">d' + r.dev + '</span>' +
            '<span style="color:' + rankColor + '"> #' + r.rank + '</span></span>';
        });
        mHtml += '</div>';
      }
      // Cumulative standings
      if (Object.keys(matchCumDevs).length > 0) {
        var sorted = Object.keys(matchCumDevs).map(function (p) {
          var pi = parseInt(p);
          var name = matchPlayers[pi] ? (matchPlayers[pi].name || 'P' + pi).substring(0, 10) : 'P' + pi;
          return { pos: pi, name: name, cumDev: matchCumDevs[pi] || 0 };
        }).sort(function (a, b) { return a.cumDev - b.cumDev; });
        mHtml += sorted.map(function (s) {
          var c = s.pos === myPos ? '#6dd5fa' : '#666';
          return '<span style="color:' + c + '">' + s.name + ':' + s.cumDev + '</span>';
        }).join('  ');
      }
      matchEl.innerHTML = mHtml || '<span style="color:#444">No match data</span>';
    }

    // Info
    var infoEl = document.getElementById('yb-info');
    if (infoEl) {
      var total = roundScores.reduce(function (a, b) { return a + b; }, 0);
      infoEl.innerHTML = 'Hand: ' + myHand.length + ' | Round pts: ' + total + '/157';
    }
  }

  // ── Init ──
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createUI);
  } else {
    createUI();
  }
})();
