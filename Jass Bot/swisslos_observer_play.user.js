// ==UserScript==
// @name         Jass Bot Observer Play
// @namespace    jass-observer
// @version      1.0.0
// @description  Shows bot recommendations as overlay — YOU make the moves
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

  // ── Card System ──
  const SUIT_ID   = { H: 0, D: 1, S: 2, C: 3 };
  const SUIT_SYM  = { H: '♥', D: '♦', S: '♠', C: '♣' };
  const SUIT_CLR  = { H: '#e53935', D: '#e53935', S: '#333', C: '#333' };
  const VAL_ID    = { '6': 0, '7': 1, '8': 2, '9': 3, '10': 4, J: 5, Q: 6, K: 7, A: 8 };

  function cardDisplay(code) {
    if (!code) return '?';
    return (SUIT_SYM[code[0]] || code[0]) + code.substring(1);
  }
  function cardHtml(code) {
    const c = SUIT_CLR[code[0]] || '#333';
    return '<span style="color:' + c + ';font-weight:bold">' + cardDisplay(code) + '</span>';
  }

  // ── State ──
  let gameSocket = null;
  let myPos = -1;
  let myHand = [];
  let trumpSuit = '';
  let seqNr = 0;
  let trickNumber = 0;
  let currentTrick = null;
  let cardsInTrick = 0;
  let roundScores = [0, 0, 0, 0];
  let serverOk = false;
  let lastRecommendation = '';
  let matchInfo = '';

  // ── Bot Communication ──
  function botFetch(path, body) {
    return fetch(BOT_URL + path, {
      method: body ? 'POST' : 'GET',
      headers: body ? { 'Content-Type': 'application/json' } : {},
      body: body ? JSON.stringify(body) : undefined,
    }).then(r => r.json());
  }

  function checkServer() {
    botFetch('/status')
      .then(() => { serverOk = true; updateUI(); })
      .catch(() => { serverOk = false; updateUI(); });
  }
  setInterval(checkServer, 5000);
  setTimeout(checkServer, 1000);

  // ── WebSocket Intercept ──
  const OrigWS = window.WebSocket;
  window.WebSocket = function (url, protocols) {
    const ws = protocols ? new OrigWS(url, protocols) : new OrigWS(url);
    if (url && url.includes('swisslos')) {
      gameSocket = ws;
      ws.addEventListener('message', function (evt) {
        try { parseMsg(evt.data); } catch (e) { console.error('[JO]', e); }
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
    const rest = raw.substring(1);
    const hashIdx = rest.indexOf('#');
    if (hashIdx < 0) return;
    const msgId = rest.substring(0, hashIdx);
    let body;
    try { body = JSON.parse(rest.substring(hashIdx + 1)); } catch { return; }
    handleMsg(msgId, body);
  }

  // ── Message Handlers ──
  function handleMsg(id, body) {
    switch (id) {
      case '9022': onGameInit(body); break;
      case '1': onCardsDealt(body); break;
      case '5': onPlayable(body); break;
      case '7': onCardPlayed(body); break;
      case '8': onTrickWon(body); break;
      case '10': case '17': onResult(body); break;
      case '19': onTrump(body); break;
    }
  }

  function onGameInit(d) {
    if (!d) return;
    trickNumber = 0;
    currentTrick = null;
    cardsInTrick = 0;
    roundScores = [0, 0, 0, 0];

    // Find our seat
    if (d.players && Array.isArray(d.players)) {
      d.players.forEach((p, i) => {
        if (p && (p.uid === null || (p.name && p.name.includes('Gast')))) myPos = p.pos !== undefined ? p.pos : i;
      });
      matchInfo = d.players.map(p => (p.name || '?').substring(0, 10)).join(' vs ');
    }

    // Trump
    if (d.trump && SUIT_ID[d.trump] !== undefined) trumpSuit = d.trump;

    // Hand
    if (d.cards && Array.isArray(d.cards)) {
      myHand = d.cards.slice();

      if (d.selectScreen && serverOk) {
        // Ask bot for declaration
        botFetch('/new-round', { hand: myHand, trump: trumpSuit, my_pos: myPos })
          .then(resp => {
            const decl = resp.declaration !== undefined ? resp.declaration : '?';
            showRecommendation('DECLARE', '🎯 Declare: <b>' + decl + '</b> points', '#2196F3');
          })
          .catch(e => showRecommendation('ERROR', 'Server error: ' + e.message, '#e53935'));
      }
    }
    updateUI();
  }

  function onCardsDealt(d) {
    if (d && d.cards) myHand = d.cards.slice();
  }

  function onTrump(d) {
    if (d) {
      const t = d.trump || d.t;
      if (t && SUIT_ID[t] !== undefined) trumpSuit = t;
    }
  }

  function onPlayable(body) {
    if (!body || !body.pCards || !body.pCards.length) return;
    const pCards = body.pCards;

    if (!serverOk) {
      showRecommendation('OFFLINE', 'Server not connected', '#e53935');
      return;
    }

    // Build trick info
    const trickCards = [];
    if (currentTrick && currentTrick.cards) {
      currentTrick.cards.forEach(c => trickCards.push({ player: c.player, card: c.card }));
    }
    const leader = currentTrick ? currentTrick.leader : 0;
    const oppPoints = [];
    for (let s = 0; s < 4; s++) {
      if (s !== myPos) oppPoints.push(roundScores[s] || 0);
    }

    botFetch('/play', {
      hand: myHand, trick_cards: trickCards,
      leader: leader, legal_moves: pCards, opp_points: oppPoints,
    }).then(resp => {
      const card = resp.card_code || pCards[0];
      const legal = pCards.map(c => (c === card ? '<u>' + cardHtml(c) + '</u>' : cardHtml(c))).join('  ');
      showRecommendation('PLAY',
        '▶ Play: <b style="font-size:28px">' + cardHtml(card) + '</b><br>' +
        '<small>Legal: ' + legal + '</small>',
        '#4CAF50');
    }).catch(e => {
      showRecommendation('ERROR', 'Play error: ' + e.message, '#e53935');
    });
  }

  function onCardPlayed(d) {
    if (!d) return;
    const pos = d.pos !== undefined ? d.pos : cardsInTrick;
    const code = d.card || d.c;
    if (!code) return;

    if (!currentTrick) {
      currentTrick = { leader: pos, cards: [] };
      cardsInTrick = 0;
    }
    if (!currentTrick.cards.some(c => c.card === code && c.player === pos)) {
      currentTrick.cards.push({ player: pos, card: code });
      cardsInTrick++;
    }

    // Remove from our hand
    const idx = myHand.indexOf(code);
    if (idx >= 0) myHand.splice(idx, 1);
  }

  function onTrickWon(d) {
    if (!d) return;
    const winner = d.pos;
    const pts = d.points || 0;
    if (winner >= 0 && winner < 4) roundScores[winner] += pts;

    // Notify server
    if (serverOk && currentTrick && currentTrick.cards.length > 0) {
      const tc = currentTrick.cards.map(c => ({ player: c.player, card: c.card }));
      botFetch('/notify-trick', { trick: tc, winner: winner, points: pts }).catch(() => {});
    }

    trickNumber++;
    currentTrick = null;
    cardsInTrick = 0;
    updateUI();
  }

  function onResult(d) {
    if (!d || !d.results) return;
    const results = d.results;
    let myRes = null;
    results.forEach(r => { if (r.pos === myPos) myRes = r; });
    if (myRes) {
      const dev = Math.abs((myRes.callP || 0) - (myRes.points || 0));
      showRecommendation('RESULT',
        'Declared: ' + myRes.callP + ' | Scored: ' + myRes.points +
        ' | <b>Dev: ' + dev + '</b>',
        dev === 0 ? '#4CAF50' : (dev <= 5 ? '#FF9800' : '#e53935'));
    }
    // End round
    if (serverOk) botFetch('/end-round', {}).catch(() => {});
  }

  // ── Recommendation Display ──
  function showRecommendation(type, html, color) {
    lastRecommendation = html;
    const el = document.getElementById('jo-rec');
    if (el) {
      el.innerHTML = html;
      el.style.borderLeftColor = color;
    }
  }

  // ── UI Overlay ──
  function createUI() {
    if (document.getElementById('jo-panel')) return;

    const panel = document.createElement('div');
    panel.id = 'jo-panel';
    panel.innerHTML =
      '<div id="jo-header">🃏 Jass Bot Observer</div>' +
      '<div id="jo-status"></div>' +
      '<div id="jo-rec" style="padding:12px;margin:8px 0;background:#111;border-radius:8px;border-left:4px solid #666;min-height:40px;font-size:16px;line-height:1.4"></div>' +
      '<div id="jo-info" style="font-size:11px;color:#888;margin-top:8px"></div>';

    panel.style.cssText =
      'position:fixed;top:10px;right:10px;width:320px;z-index:99999;' +
      'background:#1a1a2e;color:#eee;border-radius:12px;padding:14px;' +
      'font-family:system-ui,sans-serif;box-shadow:0 4px 24px rgba(0,0,0,0.5);' +
      'cursor:move;user-select:none';

    // Drag
    let dx = 0, dy = 0, mx = 0, my = 0;
    panel.querySelector('#jo-header').style.cssText =
      'font-size:15px;font-weight:bold;margin-bottom:8px;padding-bottom:6px;border-bottom:1px solid #333';
    panel.onmousedown = function (e) {
      if (e.target.tagName === 'BUTTON') return;
      mx = e.clientX; my = e.clientY;
      document.onmousemove = function (ev) {
        dx = mx - ev.clientX; dy = my - ev.clientY;
        mx = ev.clientX; my = ev.clientY;
        panel.style.top = (panel.offsetTop - dy) + 'px';
        panel.style.left = (panel.offsetLeft - dx) + 'px';
        panel.style.right = 'auto';
      };
      document.onmouseup = function () { document.onmousemove = null; };
    };

    document.body.appendChild(panel);
  }

  function updateUI() {
    if (!document.getElementById('jo-panel')) createUI();
    const statusEl = document.getElementById('jo-status');
    if (statusEl) {
      const dot = serverOk ? '🟢' : '🔴';
      statusEl.innerHTML = dot + ' Server: ' + (serverOk ? 'Connected' : 'Offline') +
        ' | Seat: ' + myPos + ' | Trump: ' + (SUIT_SYM[trumpSuit] || '-');
    }
    const infoEl = document.getElementById('jo-info');
    if (infoEl) {
      infoEl.innerHTML = matchInfo + '<br>Scores: ' + roundScores.join(' | ') +
        ' | Hand: ' + myHand.length + ' cards | Trick: ' + (trickNumber + 1);
    }
  }

  // ── Init ──
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createUI);
  } else {
    createUI();
  }
})();
