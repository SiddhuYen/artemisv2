// ══════════════════════════════════════════════════════
// BACKEND WIRING — boards/pages/contacts live on the real
// FastAPI backend (do not alter backend). Local-only bits
// (contact-to-contact links, contact photos) have no backend
// endpoint, so they're layered on top via localStorage.
// ══════════════════════════════════════════════════════
const GRAPH_ID = (() => {
  let id = localStorage.getItem('artemis_graph_id');
  if (!id) {
    id = (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : 'g'+Date.now()+Math.floor(Math.random()*1e6);
    localStorage.setItem('artemis_graph_id', id);
  }
  return id;
})();
const API_HEADERS = { 'Content-Type': 'application/json', 'X-Graph-Id': GRAPH_ID };

// A session cookie can expire while this tab sits open, after which every
// request 401s. Catching that centrally sends the user to the sign-in page
// once, instead of surfacing "authentication required" as a fake failure from
// whichever of the ~30 call sites happened to fire first.
(function guardSession() {
  const rawFetch = window.fetch.bind(window);
  window.fetch = async (...args) => {
    const res = await rawFetch(...args);
    if (res.status === 401 && !location.pathname.startsWith('/login')) {
      location.href = '/login';
    }
    return res;
  };
})();

function operatorName() { return localStorage.getItem('artemis_operator_name') || 'OPERATOR'; }
function setOperatorName(name) {
  localStorage.setItem('artemis_operator_name', name);
  ['hvBadge','ctBadge','drBadge'].forEach(id => { const el=document.getElementById(id); if(el) el.textContent = name.slice(0,2).toUpperCase(); });
  ['hvUserName','ctUserName','drUserName'].forEach(id => { const el=document.getElementById(id); if(el) el.textContent = name.toUpperCase(); });
}
function renameOperator() {
  const next = (prompt('Operator name:', operatorName()) || '').trim();
  if (next) setOperatorName(next);
}
function doLogout() {
  if (!confirm("Reset identity? This starts a fresh anonymous session on this browser — your boards and contacts stay on the server under your old id, but this browser won't see them anymore.")) return;
  localStorage.removeItem('artemis_graph_id');
  localStorage.removeItem('artemis_operator_name');
  location.reload();
}

// ── local-only enrichment layer (no backend endpoint exists for these) ──
function _localLinks() { try { return JSON.parse(localStorage.getItem('artemis_contact_links')||'{}'); } catch(e){ return {}; } }
function _saveLocalLinks(o) { localStorage.setItem('artemis_contact_links', JSON.stringify(o)); }
function _localPhotos() { try { return JSON.parse(localStorage.getItem('artemis_contact_photos')||'{}'); } catch(e){ return {}; } }
function _saveLocalPhotos(o) { localStorage.setItem('artemis_contact_photos', JSON.stringify(o)); }

// ── translation: backend node/edge (opaque JSON) <-> board person/conn ──
function personToBackendNode(p) {
  return { data: { id:p.id, label:p.name, kind:'person', role:p.role||'', company:p.company||'',
    photo:p.photo||'', description:p.description||'', size:p.size||1 },
    position: { x:p.x, y:p.y } };
}
function backendNodeToPerson(n) {
  const d = n.data || {};
  return { id:d.id, name:d.label, role:d.role||d.title||'', company:d.company||d.org||'',
    photo:d.photo||'', description:d.description||'', size:d.size||1,
    x:(n.position&&n.position.x)||0, y:(n.position&&n.position.y)||0 };
}
function connToBackendEdge(c) {
  return { data: { id:c.id, source:c.from, target:c.to, type:c.label||'', manual:true } };
}
function backendEdgeToConn(e) {
  const d = e.data || {};
  return { id:d.id, from:d.source, to:d.target, label:d.type||'' };
}
function boardSummaryToLocal(b) {
  const prev = b.preview_elements || {};
  return {
    id: b.id, name: b.name, targetName: b.target_name || '–', targetOrg: b.target_org || '',
    modified: new Date(b.created_at).getTime(), status: b.status || 'active', seq: b.seq,
    _nodeCount: b.nodes, _edgeCount: b.edges, _pageCount: b.pages,
    pages: [{ id: null, name: 'preview',
      people: (prev.nodes||[]).map(backendNodeToPerson),
      conns: (prev.edges||[]).map(backendEdgeToConn) }],
  };
}

let db = { boards: [], contacts: [] };

async function loadBoardsFromBackend() {
  try {
    const rows = await (await fetch('/boards', { headers: API_HEADERS })).json();
    db.boards = rows.map(boardSummaryToLocal);
  } catch (e) { console.error('Failed to load boards', e); }
}
async function loadContactsFromBackend() {
  try {
    const rows = await (await fetch('/network/profiles')).json();
    const links = _localLinks(), photos = _localPhotos();
    db.contacts = rows.map(p => ({
      id: p.id, name: p.canonical_name, role: (p.titles||[])[0]||'', company: (p.companies||[])[0]||'',
      email: p.email||'', photo: photos[p.id]||'', description: p.notes||'', connectedOn: p.connected_on||'',
      conns: links[p.id]||[],
    }));
  } catch (e) { console.error('Failed to load contacts', e); }
  // Single hook for the network gate: whichever path just loaded contacts
  // (boot, LinkedIn import, vCard import) closes it if there are now any.
  if (typeof syncNetworkGate === 'function') syncNetworkGate();
}

function uid() { return Math.random().toString(36).slice(2)+Date.now().toString(36); }
function esc(s) { return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function initials(name) { return name.trim().split(/\s+/).map(w=>w[0]||'').join('').slice(0,2).toUpperCase(); }
function nodeColor(id) {
  const p=['#e53e3e','#c05050','#a03030','#7f1d1d','#ef4444','#b91c1c','#dc2626','#991b1b'];
  let h=0; for(let i=0;i<id.length;i++) h=id.charCodeAt(i)+((h<<5)-h);
  return p[Math.abs(h)%p.length];
}
function isTyping() { const t=document.activeElement?.tagName; return t==='INPUT'||t==='TEXTAREA'; }
function timeAgo(ts) {
  if (!ts) return '';
  const s = Math.floor((Date.now()-ts)/1000);
  if (s < 60) return 'just now';
  if (s < 3600) return Math.floor(s/60)+'m ago';
  if (s < 86400) return Math.floor(s/3600)+'h ago';
  return Math.floor(s/86400)+'d ago';
}

let currentBoardId = null;
let currentPageId  = null;
function currentBoard() { return db.boards.find(b => b.id === currentBoardId); }
function currentPage()  {
  const b = currentBoard(); if (!b) return null;
  return b.pages.find(p => p.id === currentPageId);
}
function pageState() {
  const p = currentPage();
  return p ? { people: p.people, conns: p.conns } : { people: [], conns: [] };
}

// ══════════════════════════════════════════════════════
// HOME VIEW
// ══════════════════════════════════════════════════════
let homeFilter = 'all';

function setHomeFilter(f) {
  homeFilter = f;
  ['all','active','archived'].forEach(x => {
    const el = document.getElementById('hf'+x.charAt(0).toUpperCase()+x.slice(1));
    if(el) el.classList.toggle('on', x===f);
  });
  renderHome();
}

async function showHome() {
  document.getElementById('homeView').style.display = 'flex';
  document.getElementById('boardView').style.display = 'none';
  document.getElementById('contactsView').style.display = 'none';
  closeDetailRail();
  setOperatorName(operatorName());
  await loadBoardsFromBackend();
  renderHome();
}

function renderHome() {
  const filter = (document.getElementById('hvSearch')?.value || '').toLowerCase();
  const el = id => document.getElementById(id);

  const totalPeople = db.boards.reduce((n,b)=>n+(b._nodeCount||0),0);
  if(el('hvStatBoards')) el('hvStatBoards').textContent = db.boards.length;
  if(el('hvStatPeople')) el('hvStatPeople').textContent = totalPeople;

  const all      = db.boards.length;
  const archived = db.boards.filter(b=>(b.status||'active')==='archived').length;
  const active   = all - archived;
  if(el('hfCntAll'))      el('hfCntAll').textContent      = all;
  if(el('hfCntActive'))   el('hfCntActive').textContent   = active;
  if(el('hfCntArchived')) el('hfCntArchived').textContent = archived;

  const u = operatorName();
  if(el('hvFooterR')) el('hvFooterR').textContent = `OPERATOR ${u.toUpperCase()} · ${all} BOARD${all!==1?'S':''} · ${totalPeople} NODES INDEXED`;

  let boards = db.boards.slice().sort((a,b) => (b.modified||0)-(a.modified||0));
  if (filter) boards = boards.filter(b => {
    const tgt = b.targetName || getTargetPerson(b.pages&&b.pages[0]).name;
    return b.name.toLowerCase().includes(filter) || tgt.toLowerCase().includes(filter);
  });
  if (homeFilter === 'active')   boards = boards.filter(b => (b.status||'active') !== 'archived');
  if (homeFilter === 'archived') boards = boards.filter(b => (b.status||'active') === 'archived');

  if(el('hvCount')) el('hvCount').textContent = `[ ${boards.length} / ${all} ]`;

  const grid = el('homeGrid');
  if (!grid) return;

  if (!db.boards.length) {
    grid.innerHTML = `<div class="hv-no-match" style="padding:60px 0;grid-column:1/-1">
      <div style="font-family:var(--display-hv,'Rajdhani');font-size:36px;color:var(--ink-faint);margin-bottom:10px">ARTEMIS</div>
      No boards yet — click <span style="color:var(--accent)">NEW BOARD</span> to begin mapping.
    </div>`;
    return;
  }
  if (!boards.length) {
    grid.innerHTML = `<div class="hv-no-match" style="grid-column:1/-1">// NO BOARDS MATCH QUERY</div>`;
    return;
  }

  grid.innerHTML = boards.map((b, i) => {
    const status  = b.status || 'active';
    const brdIdx  = String(b.seq != null ? b.seq : db.boards.indexOf(b)+1).padStart(3,'0');
    const page    = b.pages && b.pages[0];
    const nodesCt = b._nodeCount ?? (b.pages ? b.pages.reduce((n,p)=>n+(p.people?.length||0),0) : 0);
    const linksCt = b._edgeCount ?? (b.pages ? b.pages.reduce((n,p)=>n+(p.conns?.length||0),0) : 0);
    const hops    = calcHops(page);
    const target  = b.targetName && b.targetName !== '–' ? { name: b.targetName, org: b.targetOrg||'' } : getTargetPerson(page);
    const upd     = timeAgo(b.modified);
    const statusLabel = status === 'priority' ? 'PRIORITY' : status === 'archived' ? 'ARCHIVED' : 'ACTIVE';
    const mm      = generateMinimap(b);

    return `<div class="hv-card fr" data-id="${b.id}" onclick="selectCard('${b.id}')" style="animation-delay:${(0.05+i*0.06).toFixed(2)}s">
      <span class="br tl"></span><span class="br tr"></span><span class="br bl"></span><span class="br br2"></span>
      <div class="hv-card-top">
        <span class="hv-pill ${status}"><span class="dot"></span>${statusLabel}</span>
        <span class="hv-brd-id">BRD-${brdIdx}</span>
      </div>
      <h3>${esc(b.name||'Untitled')}</h3>
      <div class="hv-target-line">TARGET // <b>${esc(target.name)}</b>${target.org?' · '+esc(target.org):''}</div>
      <div class="hv-preview">
        <div class="hv-pgrid"></div>
        ${mm}
      </div>
      <div class="hv-card-stats">
        <div class="s"><span class="k">NODES</span><span class="v">${nodesCt}</span></div>
        <div class="s"><span class="k">LINKS</span><span class="v">${linksCt}</span></div>
        <div class="s hop"><span class="k">HOPS</span><span class="v">${hops||'–'}</span></div>
        <div class="s"><span class="k">UPD</span><span class="v">${upd||'–'}</span></div>
        <span class="enter">ENTER ▸</span>
      </div>
    </div>`;
  }).join('') + `<div class="hv-ghost fr" onclick="showCreateModal()">
    <span class="br tl"></span><span class="br tr"></span><span class="br bl"></span><span class="br br2"></span>
    <div class="hv-ghost-in">
      <div class="hv-ghost-ring">+</div>
      <div class="hv-ghost-lbl">NEW BOARD</div>
    </div>
  </div>`;
}

function getTargetPerson(page) {
  if (!page || !page.people || !page.people.length) return { name: '–', org: '' };
  if (!page.conns || !page.conns.length) {
    const p = page.people[0]; return { name: p.name, org: p.role||'' };
  }
  const cnt = {};
  page.people.forEach(p => cnt[p.id]=0);
  page.conns.forEach(c => { cnt[c.from]=(cnt[c.from]||0)+1; cnt[c.to]=(cnt[c.to]||0)+1; });
  const top = page.people.reduce((a,b) => (cnt[b.id]||0) > (cnt[a.id]||0) ? b : a, page.people[0]);
  return { name: top.name, org: top.role||'' };
}

function calcHops(page) {
  if (!page || !page.people || page.people.length < 2 || !page.conns || !page.conns.length) return 0;
  const cnt = {};
  page.people.forEach(p => cnt[p.id]=0);
  page.conns.forEach(c => { cnt[c.from]=(cnt[c.from]||0)+1; cnt[c.to]=(cnt[c.to]||0)+1; });
  const start = page.people.reduce((a,b)=>(cnt[b.id]||0)>(cnt[a.id]||0)?b:a, page.people[0]).id;
  const adj = {}; page.people.forEach(p=>adj[p.id]=[]);
  page.conns.forEach(c=>{ adj[c.from]?.push(c.to); adj[c.to]?.push(c.from); });
  const dist = {[start]:0}; const q=[start];
  while(q.length){ const n=q.shift(); (adj[n]||[]).forEach(nb=>{ if(!(nb in dist)){ dist[nb]=dist[n]+1; q.push(nb); } }); }
  return Math.max(...Object.values(dist), 0);
}

function generateMinimap(board) {
  const page = board.pages && board.pages[0];
  if (!page || !page.people || !page.people.length) {
    return `<svg viewBox="0 0 100 56" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg"></svg>`;
  }
  const W=100, H=56, PAD=9;
  const people = page.people;

  const xs=people.map(p=>p.x), ys=people.map(p=>p.y);
  const minX=Math.min(...xs), maxX=Math.max(...xs);
  const minY=Math.min(...ys), maxY=Math.max(...ys);
  const rX=maxX-minX||1, rY=maxY-minY||1;
  const scale=Math.min((W-PAD*2)/rX,(H-PAD*2)/rY, 1);
  const ox=PAD+(W-PAD*2-rX*scale)/2, oy=PAD+(H-PAD*2-rY*scale)/2;
  const nx=p=>(ox+(p.x-minX)*scale).toFixed(1);
  const ny=p=>(oy+(p.y-minY)*scale).toFixed(1);

  const cnt={};
  people.forEach(p=>cnt[p.id]=0);
  (page.conns||[]).forEach(c=>{ cnt[c.from]=(cnt[c.from]||0)+1; cnt[c.to]=(cnt[c.to]||0)+1; });
  const sorted=[...people].sort((a,b)=>(cnt[b.id]||0)-(cnt[a.id]||0));
  const targetId=sorted[0]?.id;
  const hubCt=Math.max(1,Math.ceil(people.length*0.25));
  const hubIds=new Set(sorted.slice(1,1+hubCt).map(p=>p.id));
  const kindOf=p=>p.id===targetId?'target':hubIds.has(p.id)?'hub':'node';

  const edges=(page.conns||[]).map(c=>{
    const f=people.find(p=>p.id===c.from), t=people.find(p=>p.id===c.to);
    if(!f||!t) return '';
    const x1=+nx(f),y1=+ny(f),x2=+nx(t),y2=+ny(t);
    const dx=x2-x1,dy=y2-y1,len=Math.hypot(dx,dy)||1;
    const off=Math.min(8,len*0.18);
    const mx=((x1+x2)/2-(dy/len)*off).toFixed(1);
    const my=((y1+y2)/2+(dx/len)*off).toFixed(1);
    const hot=(f.id===targetId||t.id===targetId)?' hot':'';
    return `<path d="M${x1},${y1} Q${mx},${my} ${x2},${y2}" class="mm-edge${hot}"/>`;
  }).join('');

  const nodes=people.map(p=>{
    const cx=nx(p),cy=ny(p),kind=kindOf(p);
    const r=kind==='target'?4:kind==='hub'?3:2.2;
    if(kind==='target') return `<g transform="translate(${cx},${cy})"><circle r="7" class="mm-ring"/><circle r="7" class="mm-ping"/><circle r="${r}" class="mm-dot-target"/></g>`;
    return `<circle cx="${cx}" cy="${cy}" r="${r}" class="mm-dot-${kind}"/>`;
  }).join('');

  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid slice" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">${edges}${nodes}</svg>`;
}

async function toggleArchive(id) {
  const b = db.boards.find(x=>x.id===id); if(!b) return;
  const newStatus = (b.status==='archived') ? 'active' : 'archived';
  try {
    await fetch(`/boards/${id}`, { method:'PATCH', headers:API_HEADERS, body:JSON.stringify({status:newStatus}) });
    b.status = newStatus; b.modified = Date.now(); renderHome();
  } catch(e) { alert('Failed: '+e); }
}

// ── CARD SELECTION + DETAIL RAIL ──
let selectedCardId = null;

function selectCard(id) {
  selectedCardId = id;
  document.querySelectorAll('#homeGrid .hv-card').forEach(el =>
    el.classList.toggle('sel', el.dataset.id === id)
  );
  showDetailRail(id);
}

function showDetailRail(id) {
  const b = db.boards.find(x=>x.id===id);
  if (!b) return;
  const railEl = document.getElementById('hvRail');
  const scrim  = document.getElementById('hvRailScrim');
  if (!railEl || !scrim) return;

  const page    = b.pages && b.pages[0];
  const brdIdx  = String(b.seq != null ? b.seq : db.boards.indexOf(b)+1).padStart(3,'0');
  const nodesCt = b._nodeCount ?? (b.pages ? b.pages.reduce((n,p)=>n+(p.people?.length||0),0) : 0);
  const linksCt = b._edgeCount ?? (b.pages ? b.pages.reduce((n,p)=>n+(p.conns?.length||0),0) : 0);
  const hops    = calcHops(page);
  const target  = b.targetName && b.targetName !== '–' ? { name: b.targetName, org: b.targetOrg||'' } : getTargetPerson(page);
  const upd     = timeAgo(b.modified);

  document.getElementById('hvrCode').textContent = `// BOARD ${brdIdx}`;
  document.getElementById('hvrTitle').textContent = b.name || 'Untitled';
  document.getElementById('hvrSub').innerHTML = `TARGET // <b>${esc(target.name)}</b>${target.org?' · '+esc(target.org):''}`;

  const prev = document.getElementById('hvrPreview');
  if (prev) {
    const old = prev.querySelector('svg'); if(old) old.remove();
    prev.insertAdjacentHTML('beforeend', generateMinimap(b));
  }

  document.getElementById('hvrMetrics').innerHTML = `
    <div class="m"><div class="k">PEOPLE</div><div class="v">${nodesCt}</div></div>
    <div class="m"><div class="k">CONNECTIONS</div><div class="v">${linksCt}</div></div>
    <div class="m"><div class="k">SHORTEST PATH</div><div class="v"><b>${hops||'–'}</b>${hops?' HOP'+(hops!==1?'S':''):''}</div></div>
    <div class="m"><div class="k">LAST UPDATE</div><div class="v">${upd||'–'}</div></div>`;

  const contacts = page && page.people ? (() => {
    const c={};
    page.people.forEach(p=>c[p.id]=0);
    (page.conns||[]).forEach(x=>{c[x.from]=(c[x.from]||0)+1;c[x.to]=(c[x.to]||0)+1;});
    return [...page.people].sort((a,z)=>(c[z.id]||0)-(c[a.id]||0)).slice(0,4);
  })() : [];
  document.getElementById('hvrPath').innerHTML = contacts.length
    ? contacts.map((p,i)=>`
        <div class="hvr-pnode ${i===0?'k-target':''}">
          <div class="pip">${i===0?'●':i}</div>
          <div class="pinfo"><div class="n">${esc(p.name)}</div><div class="r">${esc(p.role||'–')}</div></div>
        </div>`).join('')
    : `<div style="font-size:11px;color:var(--ink-faint,#666)">// No contacts mapped yet</div>`;

  document.getElementById('hvrBrief').textContent =
    `${nodesCt} contact${nodesCt!==1?'s':''} mapped across ${b._pageCount||1} page${(b._pageCount||1)!==1?'s':''}. Open the board to explore and add connections.`;

  document.getElementById('hvrEnter').onclick = () => { closeDetailRail(); openBoard(id); };

  railEl.classList.add('open');
  scrim.classList.add('open');
}

function closeDetailRail() {
  const railEl = document.getElementById('hvRail');
  const scrim  = document.getElementById('hvRailScrim');
  if (railEl) railEl.classList.remove('open');
  if (scrim)  scrim.classList.remove('open');
  document.querySelectorAll('#homeGrid .hv-card').forEach(el => el.classList.remove('sel'));
  selectedCardId = null;
}

// ── CREATE MODAL ──
function showCreateModal() {
  const scrim = document.getElementById('hvModalScrim');
  if (!scrim) return;
  ['hvmName','hvmStart','hvmTarget'].forEach(id => {
    const el = document.getElementById(id); if(el) el.value='';
  });
  scrim.classList.add('open');
  setTimeout(() => document.getElementById('hvmName')?.focus(), 60);
}

function closeCreateModal() {
  const scrim = document.getElementById('hvModalScrim');
  if (scrim) scrim.classList.remove('open');
}

function hvModalScrimClick(e) {
  if (e.target === document.getElementById('hvModalScrim')) closeCreateModal();
}

async function submitCreateBoard() {
  const name   = (document.getElementById('hvmName')?.value||'').trim();
  if (!name) { document.getElementById('hvmName')?.focus(); return; }
  const start  = (document.getElementById('hvmStart')?.value||'').trim();
  const target = (document.getElementById('hvmTarget')?.value||'').trim();
  if (!start || !target) { alert('Enter both a starting person and a target.'); return; }

  try {
    const res = await fetch('/boards', { method:'POST', headers:API_HEADERS,
      body: JSON.stringify({ name: name.toUpperCase(), target_name: target.toUpperCase(), target_org: '' }) });
    if (!res.ok) throw new Error(await res.text());
    const summary = await res.json();
    closeCreateModal();
    await openBoard(summary.id);

    // designate the starting/target people straight away, same as Discover
    const pg = currentPage();
    if (pg) {
      const w = document.getElementById('wrapper');
      const cx = (w.clientWidth/2 - panX) / zoom, cy = (w.clientHeight/2 - panY) / zoom;
      const startP = findOrCreatePerson(pg, start, cx - 260, cy);
      const targetP = findOrCreatePerson(pg, target, cx + 260, cy);
      pg.startPersonId = startP.id;
      pg.targetPersonId = targetP.id;
      save(); render(); fitToContent();
    }
  } catch(e) { alert('Failed to create board: '+e.message); }
}

function newBoard() { showCreateModal(); }

async function openBoard(id) {
  let b = db.boards.find(x=>x.id===id);
  try {
    const full = await (await fetch(`/boards/${id}`, { headers: API_HEADERS })).json();
    const pages = (full.pages||[]).map(pg => ({
      id: pg.id, name: pg.name,
      people: ((pg.elements&&pg.elements.nodes)||[]).map(backendNodeToPerson),
      conns: ((pg.elements&&pg.elements.edges)||[]).map(backendEdgeToConn),
      startPersonId: (pg.elements && pg.elements.startPersonId) || null,
      targetPersonId: (pg.elements && pg.elements.targetPersonId) || null,
    }));
    if (!b) { b = { id: full.id }; db.boards.unshift(b); }
    b.name = full.name; b.targetName = full.target_name || '–'; b.targetOrg = full.target_org || '';
    b.status = full.status || 'active'; b.modified = new Date(full.created_at).getTime();
    b.pages = pages.length ? pages : [{ id: null, name: 'Page 1', people: [], conns: [] }];
  } catch(e) { alert('Failed to open board: '+e.message); return; }

  currentBoardId = id;
  currentPageId = b.pages[0].id;
  document.getElementById('homeView').style.display = 'none';
  document.getElementById('boardView').style.display = 'flex';
  document.getElementById('boardNameInput').value = b.name || 'Untitled Board';
  renderPageBar();
  render();
  fitToContent();
}

async function goHome() {
  try { closeDetail(); } catch(e) {}
  try { exitMode(); } catch(e) {}
  await flushPageSave();
  try { document.getElementById("boardView").style.display="none"; } catch(e) {}
  try { document.getElementById("contactsView").style.display="none"; } catch(e) {}
  showHome();
}

async function saveBoardName(name) {
  const b = currentBoard(); if (!b) return;
  const trimmed = name.trim() || 'Untitled Board';
  b.name = trimmed; b.modified = Date.now();
  try { await fetch(`/boards/${b.id}`, { method:'PATCH', headers:API_HEADERS, body:JSON.stringify({name:trimmed}) }); } catch(e) {}
}

async function renameBoard(id) {
  const b = db.boards.find(x=>x.id===id); if (!b) return;
  const name = prompt('Rename board:', b.name);
  if (name === null) return;
  const trimmed = name.trim() || b.name;
  b.name = trimmed; b.modified = Date.now();
  try { await fetch(`/boards/${id}`, { method:'PATCH', headers:API_HEADERS, body:JSON.stringify({name:trimmed}) }); } catch(e) {}
  renderHome();
}

async function deleteBoardConfirm(id) {
  const b = db.boards.find(x=>x.id===id); if (!b) return;
  if (!confirm(`Delete "${b.name}"? This cannot be undone.`)) return;
  try { await fetch(`/boards/${id}`, { method:'DELETE', headers:API_HEADERS }); } catch(e) {}
  db.boards = db.boards.filter(x=>x.id!==id);
  renderHome();
}

// ══════════════════════════════════════════════════════
// PAGE BAR
// ══════════════════════════════════════════════════════
function renderPageBar() {
  const b = currentBoard(); if (!b) return;
  const bar = document.getElementById('pageBar');
  bar.innerHTML = b.pages.map(pg => `
    <div class="page-tab${pg.id===currentPageId?' active':''}" onclick="switchPage('${pg.id}')">
      <span ondblclick="renamePage('${pg.id}',this)" title="Double-click to rename">${esc(pg.name)}</span>
      <span class="page-tab-x" onclick="deletePage(event,'${pg.id}')">✕</span>
    </div>
  `).join('') + `<button class="page-add-btn" onclick="addPage()" title="Add page">+</button>`;
}

async function switchPage(pgId) {
  if (pgId === currentPageId) return;
  await flushPageSave();
  currentPageId = pgId;
  renderPageBar();
  render();
  fitToContent();
}

async function addPage() {
  const b = currentBoard(); if (!b) return;
  try {
    const res = await fetch(`/boards/${b.id}/pages`, { method:'POST', headers:API_HEADERS, body:JSON.stringify({}) });
    const pg = await res.json();
    b.pages.push({ id: pg.id, name: pg.name, people: [], conns: [] });
    b.modified = Date.now();
    currentPageId = pg.id;
    renderPageBar(); render(); resetView();
  } catch(e) { alert('Failed to add page: '+e.message); }
}

async function renamePage(pgId, el) {
  const b = currentBoard(); if (!b) return;
  const pg = b.pages.find(p=>p.id===pgId); if (!pg) return;
  const name = prompt('Page name:', pg.name);
  if (name === null) return;
  const trimmed = name.trim() || pg.name;
  pg.name = trimmed;
  el.textContent = pg.name;
  b.modified = Date.now();
  try { await fetch(`/boards/${b.id}/pages/${pgId}`, { method:'PATCH', headers:API_HEADERS, body:JSON.stringify({name:trimmed}) }); } catch(e) {}
}

async function deletePage(e, pgId) {
  e.stopPropagation();
  const b = currentBoard(); if (!b) return;
  if (b.pages.length <= 1) { alert("Can't delete the only page."); return; }
  const pg = b.pages.find(p=>p.id===pgId);
  if (!confirm(`Delete "${pg?.name}"?`)) return;
  try { await fetch(`/boards/${b.id}/pages/${pgId}`, { method:'DELETE', headers:API_HEADERS }); } catch(err) {}
  b.pages = b.pages.filter(p=>p.id!==pgId);
  if (currentPageId === pgId) currentPageId = b.pages[0].id;
  b.modified = Date.now();
  renderPageBar(); render();
}

// ══════════════════════════════════════════════════════
// SAVE — debounced PATCH of the current page's elements.
// Every existing call site just calls save(); the network
// write happens automatically a moment later.
// ══════════════════════════════════════════════════════
let pageSaveTimer = null;

function save() {
  const b = currentBoard(), pg = currentPage(); if (!b||!pg) return;
  b.modified = Date.now();
  clearTimeout(pageSaveTimer);
  pageSaveTimer = setTimeout(flushPageSave, 700);
}

async function flushPageSave() {
  clearTimeout(pageSaveTimer); pageSaveTimer = null;
  const b = currentBoard(), pg = currentPage(); if (!b||!pg||!pg.id) return;
  const elements = { nodes: pg.people.map(personToBackendNode), edges: pg.conns.map(connToBackendEdge),
    startPersonId: pg.startPersonId || null, targetPersonId: pg.targetPersonId || null };
  try {
    await fetch(`/boards/${b.id}/pages/${pg.id}`, { method:'PATCH', headers:API_HEADERS, body:JSON.stringify({elements}) });
  } catch(e) { /* best-effort autosave */ }
}
window.addEventListener('beforeunload', () => { if (pageSaveTimer) flushPageSave(); });

// ══════════════════════════════════════════════════════
// ZOOM / PAN ENGINE
// ══════════════════════════════════════════════════════
let zoom=0.75, panX=0, panY=0;
let spaceDown=false, isPanning=false;
let panAnchorX=0,panAnchorY=0,panOriginX=0,panOriginY=0;
let mode='normal', connSrc=null, detailId=null, activeConn=null, dragMoved=false;
let photoTab='paste', capturedImg=null;
let selectedIds = new Set();

function toWorld(cx,cy){const r=document.getElementById('wrapper').getBoundingClientRect();return{x:(cx-r.left-panX)/zoom,y:(cy-r.top-panY)/zoom};}
function applyTransform(){const c=document.getElementById('canvas');c.style.transform=`translate(${panX}px,${panY}px) scale(${zoom})`;document.getElementById('zoomPct').textContent=Math.round(zoom*100)+'%';}
function doZoom(f,cx,cy){const r=document.getElementById('wrapper').getBoundingClientRect(),mx=cx-r.left,my=cy-r.top,nz=Math.min(6,Math.max(0.05,zoom*f));panX=mx-(mx-panX)*(nz/zoom);panY=my-(my-panY)*(nz/zoom);zoom=nz;applyTransform();}
function changeZoom(f){const r=document.getElementById('wrapper').getBoundingClientRect();doZoom(f,r.left+r.width/2,r.top+r.height/2);}
function resetView(){const w=document.getElementById('wrapper');zoom=0.75;panX=w.clientWidth/2-10000*zoom/2;panY=w.clientHeight/2-7000*zoom/2;applyTransform();}
function fitToContent(){
  const st=pageState();
  const w=document.getElementById('wrapper');
  if(!st.people.length){resetView();return;}
  const PAD=160;
  const xs=st.people.map(p=>p.x), ys=st.people.map(p=>p.y);
  const minX=Math.min(...xs)-PAD, maxX=Math.max(...xs)+PAD;
  const minY=Math.min(...ys)-PAD, maxY=Math.max(...ys)+PAD;
  const scaleX=w.clientWidth/(maxX-minX), scaleY=w.clientHeight/(maxY-minY);
  zoom=Math.min(scaleX,scaleY,1.4);
  zoom=Math.max(zoom,0.08);
  panX=w.clientWidth/2-((minX+maxX)/2)*zoom;
  panY=w.clientHeight/2-((minY+maxY)/2)*zoom;
  applyTransform();
}

document.getElementById('wrapper').addEventListener('wheel',e=>{e.preventDefault();Math.abs(e.deltaX)>Math.abs(e.deltaY)*1.5&&!e.ctrlKey?(panX-=e.deltaX,panY-=e.deltaY,applyTransform()):doZoom(e.deltaY<0?1.09:0.91,e.clientX,e.clientY);},{passive:false});

document.addEventListener('keydown',e=>{
  if(e.code==='Space'&&!isTyping()){e.preventDefault();if(!spaceDown){spaceDown=true;document.getElementById('wrapper').classList.add('space-pan');}}
  if(isTyping()) return;
  switch(e.key){
    case'Escape':
      if(selectedIds.size>0){selectedIds.clear();renderMultiSel();break;}
      if(document.getElementById('liScrim')?.classList.contains('open')){closeLinkedInImport();break;}
      if(document.getElementById('discoverScrim')?.classList.contains('open')){closeDiscoverModal();break;}
      if(mode!=='normal')exitMode();else closeDetail();break;
    case'c':case'C':toggleMode('connect');break;
    case'd':case'D':toggleMode('delete');break;
    case'a':case'A':openAddModal();break;
    case'+':case'=':changeZoom(1.2);break;
    case'-':changeZoom(0.8);break;
    case'0':resetView();break;
    case'Delete':case'Backspace':if(selectedIds.size>1&&mode==='normal'){const pg=currentPage();const toKill=[...selectedIds];pg.people=pg.people.filter(p=>!toKill.includes(p.id));pg.conns=pg.conns.filter(c=>!toKill.includes(c.from)&&!toKill.includes(c.to));selectedIds.clear();save();render();}break;
  }
});
document.addEventListener('keyup',e=>{if(e.code==='Space'){spaceDown=false;isPanning=false;document.getElementById('wrapper').classList.remove('space-pan','panning');}});

const wrapper=document.getElementById('wrapper');
wrapper.addEventListener('mousedown',e=>{
  const onNode=e.target.closest('.node')||e.target.closest('.resize-handle');
  if(e.button===1||(e.button===0&&!onNode)){e.preventDefault();isPanning=true;panAnchorX=e.clientX;panAnchorY=e.clientY;panOriginX=panX;panOriginY=panY;wrapper.classList.add('panning');}
});
document.addEventListener('mousemove',e=>{if(isPanning){panX=panOriginX+(e.clientX-panAnchorX);panY=panOriginY+(e.clientY-panAnchorY);applyTransform();}});
document.addEventListener('mouseup',e=>{if(e.button===1||(isPanning&&!spaceDown)){isPanning=false;wrapper.classList.remove('panning');}});

wrapper.addEventListener('mousemove',e=>{
  const r=wrapper.getBoundingClientRect();
  wrapper.style.setProperty('--vmx',(e.clientX-r.left)+'px');
  wrapper.style.setProperty('--vmy',(e.clientY-r.top)+'px');
  if(mode==='connect'&&connSrc){
    const w=toWorld(e.clientX,e.clientY);
    const src=pageState().people.find(p=>p.id===connSrc);if(!src)return;
    const tl=document.getElementById('tempLine');
    tl.setAttribute('x1',src.x);tl.setAttribute('y1',src.y);tl.setAttribute('x2',w.x);tl.setAttribute('y2',w.y);tl.style.display='';
  }
});
wrapper.addEventListener('mouseleave',()=>{wrapper.style.setProperty('--vmx','-9999px');wrapper.style.setProperty('--vmy','-9999px');});

// ══════════════════════════════════════════════════════
// RENDER
// ══════════════════════════════════════════════════════
function render(){
  const st=pageState();
  renderConns(st);
  renderNodes(st);
  document.getElementById('emptyState').style.display=st.people.length?'none':'';
}

function renderNodes(st) {
  const canvas=document.getElementById('canvas');
  const pg=currentPage();
  canvas.querySelectorAll('.node').forEach(n=>n.remove());
  (st||pageState()).people.forEach(p=>{
    const sz=p.size||1,avPx=Math.round(80*sz),br=Math.round(6*sz);
    const col=nodeColor(p.id),ini=initials(p.name);
    const div=document.createElement('div');
    div.className='node'+(mode==='connect'?' m-connect':'')+(mode==='delete'?' m-delete':'')+(p.id===detailId?' selected':'')+(p.id===connSrc?' conn-src':'');
    div.id=`nd-${p.id}`;div.style.cssText=`left:${p.x}px;top:${p.y}px`;
    const tagHtml = pg&&pg.startPersonId===p.id ? `<div class="node-tag node-tag-origin">POINT OF ORIGIN</div>`
      : pg&&pg.targetPersonId===p.id ? `<div class="node-tag node-tag-target">TARGET</div>` : '';
    div.innerHTML=`
      <div class="avatar-box" style="width:${avPx}px;height:${avPx}px;border-radius:${br}px;position:relative">
        ${p.photo?`<img src="${esc(p.photo)}" onerror="this.style.display='none';this.nextSibling.style.display='flex'"><div class="avatar-initials" style="background:${col};display:none">${ini}</div>`:`<div class="avatar-initials" style="background:${col}">${ini}</div>`}
        <div class="resize-handle" data-id="${esc(p.id)}"></div>
      </div>
      <div class="node-name" style="font-size:${(0.77*sz).toFixed(2)}rem;max-width:${Math.round(110*sz)}px">${esc(p.name)}</div>
      ${p.role?`<div class="node-role" style="font-size:${(0.62*sz).toFixed(2)}rem;max-width:${Math.round(110*sz)}px">${esc(p.role)}</div>`:''}
      ${tagHtml}
    `;
    div.addEventListener('mousedown',e=>nodeDown(e,p.id));
    div.addEventListener('click',e=>nodeClick(e,p.id));
    div.querySelector('.resize-handle').addEventListener('mousedown',e=>resizeDown(e,p.id));
    canvas.appendChild(div);
  });
}

function ensureConnFilter(svg) {
  if (svg.querySelector('#connGlow')) return;
  const defs = svgEl('defs');
  defs.innerHTML = `
    <filter id="connGlow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur in="SourceGraphic" stdDeviation="3" result="blur"/>
      <feColorMatrix in="blur" type="matrix"
        values="1 0 0 0 0.9  0 0 0 0 0.1  0 0 0 0 0.1  0 0 0 1 0" result="glow"/>
      <feMerge><feMergeNode in="glow"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <marker id="arrowHead" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
      <path d="M0,0.5 L0,5.5 L5.5,3 z" fill="rgba(220,60,60,0.7)"/>
    </marker>`;
  svg.appendChild(defs);
}
function renderConns(st) {
  const svg=document.getElementById('svg');
  [...svg.children].forEach(el=>{if(el.id!=='tempLine'&&el.tagName!=='defs')el.remove();});
  ensureConnFilter(svg);
  (st||pageState()).conns.forEach(c=>{
    const people=(st||pageState()).people;
    const f=people.find(p=>p.id===c.from),t=people.find(p=>p.id===c.to);
    if(!f||!t)return;
    const x1=f.x,y1=f.y,x2=t.x,y2=t.y,mx=(x1+x2)/2,my=(y1+y2)/2;
    const dx=x2-x1,dy=y2-y1,len=Math.sqrt(dx*dx+dy*dy)||1,off=Math.min(70,len*.22);
    const cpx=mx+(-dy/len)*off,cpy=my+(dx/len)*off;
    const d=`M${x1},${y1} Q${cpx},${cpy} ${x2},${y2}`;
    const glow=svgEl('path');glow.setAttribute('d',d);glow.setAttribute('stroke','rgba(220,50,50,0.18)');glow.setAttribute('stroke-width','7');glow.setAttribute('fill','none');glow.setAttribute('pointer-events','none');glow.setAttribute('filter','url(#connGlow)');
    svg.insertBefore(glow,svg.firstChild);
    const vis=svgEl('path');vis.setAttribute('d',d);vis.setAttribute('stroke','rgba(210,55,55,0.82)');vis.setAttribute('stroke-width','2');vis.setAttribute('fill','none');vis.setAttribute('pointer-events','none');vis.setAttribute('stroke-linecap','round');
    svg.insertBefore(vis,svg.firstChild);
    const hit=svgEl('path');hit.setAttribute('d',d);hit.setAttribute('stroke','transparent');hit.setAttribute('stroke-width','22');hit.setAttribute('fill','none');hit.style.cursor='pointer';hit.style.pointerEvents='stroke';
    hit.addEventListener('click',e=>connClick(e,c.id));svg.insertBefore(hit,svg.firstChild);
    if(c.label){
      const bg=svgEl('rect');
      const tw=c.label.length*6+10;
      bg.setAttribute('x',cpx-tw/2);bg.setAttribute('y',cpy-20);bg.setAttribute('width',tw);bg.setAttribute('height',16);bg.setAttribute('rx','3');bg.setAttribute('fill','rgba(12,4,4,0.85)');bg.setAttribute('stroke','rgba(180,40,40,0.4)');bg.setAttribute('stroke-width','0.8');bg.setAttribute('pointer-events','none');svg.appendChild(bg);
      const tx=svgEl('text');tx.setAttribute('x',cpx);tx.setAttribute('y',cpy-8);tx.setAttribute('text-anchor','middle');tx.setAttribute('fill','rgba(220,140,140,0.9)');tx.setAttribute('font-size','9.5');tx.setAttribute('pointer-events','none');tx.setAttribute('font-family','JetBrains Mono, monospace');tx.setAttribute('letter-spacing','0.06em');tx.textContent=c.label;svg.appendChild(tx);
    }
  });
}
function svgEl(tag){return document.createElementNS('http://www.w3.org/2000/svg',tag);}

// ══════════════════════════════════════════════════════
// NODE INTERACTION
// ══════════════════════════════════════════════════════
function nodeDown(e,id){
  if((mode!=='normal'&&mode!=='select')||spaceDown||isPanning)return;
  e.stopPropagation();
  const st=pageState();const p=st.people.find(x=>x.id===id);if(!p)return;
  const sx=e.clientX,sy=e.clientY;
  const toMove = (mode==='select'&&selectedIds.has(id)) ? [...selectedIds] : [id];
  const startPos={};
  toMove.forEach(sid=>{const sp=st.people.find(x=>x.id===sid);if(sp)startPos[sid]={x:sp.x,y:sp.y};});
  dragMoved=false;
  const mv=ev=>{
    const dx=(ev.clientX-sx)/zoom,dy=(ev.clientY-sy)/zoom;
    if(Math.abs(dx)>2||Math.abs(dy)>2)dragMoved=true;
    toMove.forEach(sid=>{
      const sp=st.people.find(x=>x.id===sid);if(!sp||!startPos[sid])return;
      sp.x=Math.max(50,startPos[sid].x+dx);sp.y=Math.max(50,startPos[sid].y+dy);
      const el=document.getElementById(`nd-${sid}`);if(el){el.style.left=sp.x+'px';el.style.top=sp.y+'px';}
    });
    renderConns();
  };
  const up=()=>{if(dragMoved)save();document.removeEventListener('mousemove',mv);document.removeEventListener('mouseup',up);};
  document.addEventListener('mousemove',mv);document.addEventListener('mouseup',up);
}

function nodeClick(e,id){
  e.stopPropagation();if(dragMoved){dragMoved=false;return;}if(isPanning)return;
  const st=pageState();
  if(mode==='delete'){const name=st.people.find(p=>p.id===id)?.name;if(!confirm(`Delete "${name}"?`))return;const pg=currentPage();pg.people=pg.people.filter(p=>p.id!==id);pg.conns=pg.conns.filter(c=>c.from!==id&&c.to!==id);if(detailId===id)closeDetail();save();render();return;}
  if(mode==='connect'){
    if(!connSrc){connSrc=id;setHint('Click another person to connect…');renderNodes();}
    else if(connSrc===id){connSrc=null;renderNodes();}
    else{const exists=st.conns.find(c=>(c.from===connSrc&&c.to===id)||(c.from===id&&c.to===connSrc));if(!exists){const pg=currentPage();pg.conns.push({id:uid(),from:connSrc,to:id,label:''});save();}connSrc=null;setHint('Click a person to connect · Esc to exit');render();}
    return;
  }
  if(e.shiftKey){
    if(selectedIds.has(id)){selectedIds.delete(id);}else{selectedIds.add(id);}
    renderMultiSel();
    return;
  }
  selectedIds.clear();renderMultiSel();
  openDetail(id);
}
function renderMultiSel(){
  document.querySelectorAll('.node.multi-sel').forEach(n=>n.classList.remove('multi-sel'));
  selectedIds.forEach(id=>{const el=document.getElementById(`nd-${id}`);if(el)el.classList.add('multi-sel');});
  if(selectedIds.size>1)setHint(`${selectedIds.size} nodes selected · Drag to move · Delete to remove · Esc to clear`);
}

function resizeDown(e,id){
  e.stopPropagation();e.preventDefault();
  const p=pageState().people.find(x=>x.id===id);if(!p)return;
  const startSize=p.size||1,sx=e.clientX,sy=e.clientY;
  const onMove=ev=>{p.size=Math.min(4,Math.max(0.25,startSize+((ev.clientX-sx)+(ev.clientY-sy))/(120*zoom)));const el=document.getElementById(`nd-${id}`);if(!el)return;const sz=p.size,avPx=Math.round(80*sz),br=Math.round(6*sz);const box=el.querySelector('.avatar-box');if(box){box.style.width=avPx+'px';box.style.height=avPx+'px';box.style.borderRadius=br+'px';}const nm=el.querySelector('.node-name'),rl=el.querySelector('.node-role');if(nm){nm.style.fontSize=(0.77*sz).toFixed(2)+'rem';nm.style.maxWidth=Math.round(110*sz)+'px';}if(rl){rl.style.fontSize=(0.62*sz).toFixed(2)+'rem';rl.style.maxWidth=Math.round(110*sz)+'px';}renderConns();const sl=document.getElementById('dp-size'),sv=document.getElementById('dp-size-val');if(sl&&detailId===id){sl.value=p.size;sv.textContent=Math.round(p.size*100);}};
  const onUp=()=>{save();document.removeEventListener('mousemove',onMove);document.removeEventListener('mouseup',onUp);};
  document.addEventListener('mousemove',onMove);document.addEventListener('mouseup',onUp);
}

// ══════════════════════════════════════════════════════
// CONNECTION POPUP
// ══════════════════════════════════════════════════════
function connClick(e,id){
  e.stopPropagation();
  if(mode==='delete'){const pg=currentPage();pg.conns=pg.conns.filter(c=>c.id!==id);save();renderConns();return;}
  activeConn=id;const c=pageState().conns.find(x=>x.id===id);
  document.getElementById('connLabelIn').value=c?.label||'';
  const pop=document.getElementById('connPopup');pop.style.left=(e.clientX-110)+'px';pop.style.top=(e.clientY-80)+'px';pop.classList.add('open');setTimeout(()=>document.getElementById('connLabelIn').focus(),30);
}
function saveConn(){const pg=currentPage();const c=pg.conns.find(x=>x.id===activeConn);if(c){c.label=document.getElementById('connLabelIn').value.trim();save();renderConns();}document.getElementById('connPopup').classList.remove('open');}
function delConn(){const pg=currentPage();pg.conns=pg.conns.filter(c=>c.id!==activeConn);save();renderConns();document.getElementById('connPopup').classList.remove('open');}
document.addEventListener('click',e=>{if(!e.target.closest('#connPopup'))document.getElementById('connPopup').classList.remove('open');});

// ══════════════════════════════════════════════════════
// DETAIL PANEL
// ══════════════════════════════════════════════════════
function openDetail(id){
  const p=pageState().people.find(x=>x.id===id);if(!p)return;
  detailId=id;renderNodes();
  const pg=currentPage();
  const col=nodeColor(p.id),ini=initials(p.name);
  const photoHTML=p.photo?`<img id="dpBigImg" src="${esc(p.photo)}" onerror="this.style.display='none'">`:`<div class="dp-initials-big" style="background:${col}">${ini}</div>`;
  document.getElementById('dpTwoCol').innerHTML=`
    <div class="dp-left-col">
      <div class="dp-big-img-wrap" id="dpBigWrap" onclick="dpZoneClick()" ondragover="event.preventDefault()" ondrop="dpDrop(event)">
        ${photoHTML}<div class="dp-img-hover">📷 CHANGE PHOTO</div>
      </div>
      <div class="dp-size-section">
        <div class="dp-size-lbl"><span>IMAGE SIZE</span><strong id="dp-size-val">${Math.round((p.size||1)*100)}</strong></div>
        <input type="range" id="dp-size" min="0.25" max="4" step="0.05" value="${(p.size||1).toFixed(2)}" oninput="liveSize(this.value)">
      </div>
      <input class="dp-photo-url" type="text" id="dp-photo" value="${esc(p.photo||'')}" placeholder="…or paste image URL" oninput="dpPhotoUrl()">
    </div>
    <div class="dp-right-col">
      <input class="dp-name-input" type="text" id="dp-name" value="${esc(p.name)}" spellcheck="false" placeholder="Name…">
      <div class="dp-tag-row">
        <button class="dp-tag-btn dp-tag-origin${pg&&pg.startPersonId===p.id?' active':''}" id="dpOriginBtn" onclick="toggleOriginTag('${p.id}')">📍 POINT OF ORIGIN</button>
        <button class="dp-tag-btn dp-tag-target${pg&&pg.targetPersonId===p.id?' active':''}" id="dpTargetBtn" onclick="toggleTargetTag('${p.id}')">🎯 TARGET</button>
      </div>
      <div class="dp-field"><label>// affiliation</label>
        <input type="text" id="dp-role" value="${esc(p.role||'')}" placeholder="Role, company, relationship…">
      </div>
      <div class="dp-field" style="flex:1"><label>// description</label>
        <textarea id="dp-desc" style="min-height:88px">${esc(p.description||'')}</textarea>
      </div>
      <div class="dp-bottom-row">
        <button class="btn-del" onclick="delFromDetail()">Delete Person</button>
        <button class="btn-save" id="dpSaveBtn" onclick="saveDetail()">SAVE</button>
      </div>
    </div>
  `;
  document.getElementById('detailPanel').classList.add('open');
  document.getElementById('dpBackdrop').classList.add('open');
}

function dpZoneClick(){if(!navigator.clipboard?.read)return;navigator.clipboard.read().then(items=>{for(const it of items){const t=it.types.find(x=>x.startsWith('image/'));if(t){it.getType(t).then(b=>fileToB64(b,url=>{document.getElementById('dp-photo').value='';dpSetPhoto(url);}));return;}}}).catch(()=>{});}
function dpDrop(e){e.preventDefault();const f=e.dataTransfer.files[0];if(f&&f.type.startsWith('image/')){fileToB64(f,url=>{document.getElementById('dp-photo').value='';dpSetPhoto(url);});return;}const url=e.dataTransfer.getData('text/plain');if(url&&url.startsWith('http')){document.getElementById('dp-photo').value=url;dpSetPhoto(url);}}
function dpPhotoUrl(){const url=document.getElementById('dp-photo').value.trim();if(url)dpSetPhoto(url);}
function dpSetPhoto(url){const w=document.getElementById('dpBigWrap');if(!w)return;let img=w.querySelector('img');if(!img){const init=w.querySelector('.dp-initials-big');if(init)init.remove();img=document.createElement('img');img.id='dpBigImg';w.insertBefore(img,w.querySelector('.dp-img-hover'));}img.src=url;img.style.display='block';}
function closeDetail(){document.getElementById('detailPanel').classList.remove('open');document.getElementById('dpBackdrop').classList.remove('open');detailId=null;renderNodes();}

function toggleOriginTag(id){
  const pg=currentPage(); if(!pg) return;
  pg.startPersonId = (pg.startPersonId===id) ? null : id;
  save(); renderNodes(); updateDetailTagButtons();
}
function toggleTargetTag(id){
  const pg=currentPage(); if(!pg) return;
  pg.targetPersonId = (pg.targetPersonId===id) ? null : id;
  save(); renderNodes(); updateDetailTagButtons();
}
function updateDetailTagButtons(){
  if(!detailId) return;
  const pg=currentPage(); if(!pg) return;
  document.getElementById('dpOriginBtn')?.classList.toggle('active', pg.startPersonId===detailId);
  document.getElementById('dpTargetBtn')?.classList.toggle('active', pg.targetPersonId===detailId);
}
function liveSize(val){if(!detailId)return;const p=pageState().people.find(x=>x.id===detailId);if(!p)return;p.size=parseFloat(val);document.getElementById('dp-size-val').textContent=Math.round(p.size*100);const el=document.getElementById(`nd-${detailId}`);if(!el)return;const sz=p.size,box=el.querySelector('.avatar-box');if(box){box.style.width=Math.round(80*sz)+'px';box.style.height=Math.round(80*sz)+'px';box.style.borderRadius=Math.round(6*sz)+'px';}const nm=el.querySelector('.node-name'),rl=el.querySelector('.node-role');if(nm){nm.style.fontSize=(0.77*sz).toFixed(2)+'rem';nm.style.maxWidth=Math.round(110*sz)+'px';}if(rl){rl.style.fontSize=(0.62*sz).toFixed(2)+'rem';rl.style.maxWidth=Math.round(110*sz)+'px';}renderConns();}
function saveDetail(){if(!detailId)return;const p=pageState().people.find(x=>x.id===detailId);if(!p)return;p.name=document.getElementById('dp-name').value.trim()||p.name;p.role=document.getElementById('dp-role').value.trim();p.description=document.getElementById('dp-desc').value;p.size=parseFloat(document.getElementById('dp-size').value)||1;const urlIn=document.getElementById('dp-photo').value.trim();const bigImg=document.querySelector('#dpBigWrap img');p.photo=bigImg?bigImg.src:(urlIn||p.photo);save();renderNodes();renderConns();const btn=document.getElementById('dpSaveBtn');if(btn){btn.textContent='✓ SAVED';setTimeout(()=>btn.textContent='SAVE',1500);}}
function delFromDetail(){if(!detailId)return;const name=pageState().people.find(p=>p.id===detailId)?.name;if(!confirm(`Delete "${name}"?`))return;const pg=currentPage();pg.people=pg.people.filter(p=>p.id!==detailId);pg.conns=pg.conns.filter(c=>c.from!==detailId&&c.to!==detailId);closeDetail();save();render();}


// ══════════════════════════════════════════════════════
// ADD MODAL
// ══════════════════════════════════════════════════════
function openAddModal(){['mName','mUrl','mRole'].forEach(id=>{const el=document.getElementById(id);if(el)el.value='';});document.getElementById('mDesc').value='';capturedImg=null;resetModalZone();switchTab('paste');document.getElementById('overlay').classList.add('open');setTimeout(()=>document.getElementById('mName').focus(),80);}
function closeModal(){document.getElementById('overlay').classList.remove('open');}
function bgClick(e){if(e.target===document.getElementById('overlay'))closeModal();}
function switchTab(t){photoTab=t;['paste','url','upload'].forEach(n=>{document.getElementById('pt'+n.charAt(0).toUpperCase()+n.slice(1)).className='ptab'+(t===n?' on':'');const s=document.getElementById('sec'+n.charAt(0).toUpperCase()+n.slice(1));if(s)s.style.display=t===n?'':'none';});}
function resetModalZone(){const z=document.getElementById('mPasteZone');if(!z)return;z.classList.remove('has-img','dragover');z.innerHTML=`<div class="paste-icon">📋</div><div class="paste-hint">Press <kbd>Ctrl+V</kbd> / <kbd>⌘V</kbd> to paste<br><span style="color:#2a1818">or drag &amp; drop an image</span></div>`;}
function setModalZoneImg(url){capturedImg=url;const z=document.getElementById('mPasteZone');if(!z)return;z.classList.add('has-img');z.innerHTML=`<img src="${esc(url)}" style="max-height:130px;object-fit:contain;border-radius:4px;max-width:100%">`;}
function modalPasteClick(){if(!navigator.clipboard?.read)return;navigator.clipboard.read().then(items=>{for(const it of items){const t=it.types.find(x=>x.startsWith('image/'));if(t){it.getType(t).then(b=>fileToB64(b,setModalZoneImg));return;}}}).catch(()=>{});}
function modalDrop(e){e.preventDefault();document.getElementById('mPasteZone').classList.remove('dragover');const f=e.dataTransfer.files[0];if(f&&f.type.startsWith('image/')){fileToB64(f,setModalZoneImg);return;}const url=e.dataTransfer.getData('text/plain');if(url&&url.startsWith('http'))setModalZoneImg(url);}
function urlPreview(){const url=document.getElementById('mUrl').value.trim();const p=document.getElementById('urlPrev');p.innerHTML=url?`<img src="${esc(url)}" style="width:100%;height:100%;object-fit:cover;border-radius:0" onerror="this.parentNode.innerHTML='<span style=\\'font-size:.7rem;color:#e53e3e;padding:4px\\'>bad url</span>'">`:'<span style="color:#9a6060;font-size:1.3rem">📷</span>';}
function fileUpload(){const f=document.getElementById('mFile').files[0];if(!f)return;fileToB64(f,url=>{capturedImg=url;const p=document.getElementById('filePrev');if(p)p.innerHTML=`<img src="${url}" style="width:100%;height:100%;object-fit:cover;border-radius:0">`;});}

document.addEventListener('paste',e=>{
  const modalOpen=document.getElementById('overlay').classList.contains('open');
  const detailOpen=document.getElementById('detailPanel').classList.contains('open');
  if(!modalOpen&&!detailOpen)return;if(isTyping())return;
  const img=Array.from(e.clipboardData?.items||[]).find(i=>i.type.startsWith('image/'));if(!img)return;
  e.preventDefault();
  fileToB64(img.getAsFile(),url=>{if(modalOpen){switchTab('paste');setModalZoneImg(url);}else{dpSetPhoto(url);document.getElementById('dp-photo').value='';}});
});

function submitAdd(){
  const name=document.getElementById('mName').value.trim();if(!name){document.getElementById('mName').focus();return;}
  let photo='';
  if(photoTab==='paste'){photo=capturedImg||'';const zi=document.querySelector('#mPasteZone img');if(zi)photo=zi.src;}
  else if(photoTab==='url'){photo=document.getElementById('mUrl').value.trim();}
  else{photo=capturedImg||'';}
  const w=document.getElementById('wrapper');
  const cx=(w.clientWidth/2-panX)/zoom,cy=(w.clientHeight/2-panY)/zoom;
  const j=()=>(Math.random()-.5)*220;
  const pg=currentPage();if(!pg)return;
  const p={id:uid(),name,role:document.getElementById('mRole').value.trim(),photo,description:document.getElementById('mDesc').value.trim(),size:1,x:Math.round(cx+j()),y:Math.round(cy+j())};
  pg.people.push(p);save();render();closeModal();
  setTimeout(()=>openDetail(p.id),120);
}

function fileToB64(file,cb){const r=new FileReader();r.onload=e=>cb(e.target.result);r.readAsDataURL(file);}

// ══════════════════════════════════════════════════════
// MODE
// ══════════════════════════════════════════════════════
function toggleMode(m){
  mode=(mode===m)?'normal':m;
  if(mode==='normal'){connSrc=null;document.getElementById('tempLine').style.display='none';}
  document.getElementById('wrapper').className='canvas-wrapper'+(mode!=='normal'?` m-${mode}`:'');
  document.getElementById('miConnect')?.classList.toggle('active', mode==='connect');
  document.getElementById('miDelete')?.classList.toggle('active', mode==='delete');
  const menuBtn = document.getElementById('btnManualMenu');
  if (menuBtn) menuBtn.className = 'btn' + (mode==='connect' ? ' m-connect' : mode==='delete' ? ' m-delete' : '');
  setHint({normal:'scroll=zoom · drag=pan · [C]onnect · [D]elete · [A]dd',connect:'click a person to start · Esc cancel',delete:'click person or line to delete · Esc cancel'}[mode]);
  renderNodes();
}
function exitMode(){mode='normal';connSrc=null;document.getElementById('tempLine').style.display='none';selectedIds.clear();toggleMode('normal');}
function setHint(t){}

function toggleManualMenu(e){
  e.stopPropagation();
  const menu = document.getElementById('manualMenu');
  const opening = !menu.classList.contains('open');
  if (opening) {
    const r = document.getElementById('btnManualMenu').getBoundingClientRect();
    menu.style.top = (r.bottom + 6) + 'px';
    menu.style.left = r.left + 'px';
  }
  menu.classList.toggle('open', opening);
}
function closeManualMenu(){
  document.getElementById('manualMenu')?.classList.remove('open');
}
document.addEventListener('click', (e) => {
  if (!e.target.closest('.tb-menu-wrap')) closeManualMenu();
});
// Discover-results "source" cards carry their URL in a data attribute (never
// inline onclick=""): an inline handler's string is HTML-entity-decoded before
// it's run as JS, so escaping quotes in esc() wouldn't stop a source_url
// (live web-search data) from breaking out of the handler's JS string.
document.addEventListener('click', (e) => {
  const card = e.target.closest('.ct-card[data-src-url]');
  if (card) window.open(card.dataset.srcUrl, '_blank');
});
function clearBoard(){if(!confirm('Clear this page?'))return;const pg=currentPage();if(!pg)return;pg.people=[];pg.conns=[];connSrc=null;save();closeDetail();render();}

// ══════════════════════════════════════════════════════
// BULK IMPORT (manual rows / CSV) — adds plain nodes to the
// current board page; unrelated to My Connections / backend.
// ══════════════════════════════════════════════════════
let importRowData = [];
let importCsvRows = [];
let importActiveTab = 'manual';

function openImportModal() {
  importRowData = [{name:'',affil:'',connTo:'',photo:''}];
  importCsvRows = [];
  importActiveTab = 'manual';
  renderImportRows();
  document.getElementById('csvStatus').textContent = '';
  document.getElementById('csvPreview').innerHTML = '';
  const fi = document.getElementById('csvFileIn'); if (fi) fi.value = '';
  switchImportTab('manual');
  document.getElementById('importOverlay').classList.add('open');
}

function closeImportModal() {
  document.getElementById('importOverlay').classList.remove('open');
}

function importBgClick(e) {
  if (e.target === document.getElementById('importOverlay')) closeImportModal();
}

function switchImportTab(tab) {
  importActiveTab = tab;
  document.getElementById('importManualSec').style.display = tab === 'manual' ? '' : 'none';
  document.getElementById('importCsvSec').style.display    = tab === 'csv'    ? '' : 'none';
  document.getElementById('itManual').className = 'ptab' + (tab === 'manual' ? ' on' : '');
  document.getElementById('itCsv').className    = 'ptab' + (tab === 'csv'    ? ' on' : '');
}

function renderImportRows() {
  const c = document.getElementById('importRowsContainer');
  c.innerHTML = importRowData.map((row, i) => `
    <div class="imp-row">
      <input class="imp-cell" type="text" placeholder="Name *"
        value="${esc(row.name)}" oninput="importRowData[${i}].name=this.value">
      <input class="imp-cell" type="text" placeholder="CEO, Investor…"
        value="${esc(row.affil)}" oninput="importRowData[${i}].affil=this.value">
      <input class="imp-cell" type="text" placeholder="Other name in batch"
        value="${esc(row.connTo)}" oninput="importRowData[${i}].connTo=this.value">
      <input class="imp-cell" type="text" placeholder="https://…"
        value="${esc(row.photo)}" oninput="importRowData[${i}].photo=this.value">
      <button class="imp-del-btn" onclick="removeImportRow(${i})">✕</button>
    </div>
  `).join('');
}

function addImportRow() {
  importRowData.push({name:'',affil:'',connTo:'',photo:''});
  renderImportRows();
  const rows = document.querySelectorAll('#importRowsContainer .imp-row');
  const last = rows[rows.length-1];
  if (last) last.querySelector('.imp-cell').focus();
  const sa = document.querySelector('.imp-scroll-area'); if (sa) sa.scrollTop = sa.scrollHeight;
}

function removeImportRow(i) {
  importRowData.splice(i, 1);
  if (!importRowData.length) importRowData = [{name:'',affil:'',connTo:'',photo:''}];
  renderImportRows();
}

function parseCSV(text) {
  const rows = [];
  const lines = text.split(/\r?\n/);
  for (const line of lines) {
    if (!line.trim()) continue;
    const cells = [];
    let cur = '', inQ = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        if (inQ && line[i+1] === '"') { cur += '"'; i++; }
        else inQ = !inQ;
      } else if (ch === ',' && !inQ) {
        cells.push(cur.trim()); cur = '';
      } else {
        cur += ch;
      }
    }
    cells.push(cur.trim());
    rows.push(cells);
  }
  return rows;
}

function detectColumns(headers) {
  const find = (...patterns) => {
    for (const p of patterns) {
      const idx = headers.findIndex(h => h.toLowerCase().includes(p));
      if (idx >= 0) return idx;
    }
    return -1;
  };
  return {
    name:   find('name', 'person', 'who', 'full'),
    affil:  find('affil', 'role', 'company', 'org', 'title', 'position'),
    connTo: find('connect', 'link', 'to', 'relation', 'assoc'),
    photo:  find('photo', 'image', 'pic', 'avatar', 'url')
  };
}

function handleCSVUpload(e) {
  const file = e.target.files[0]; if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    const rows = parseCSV(ev.target.result);
    document.getElementById('csvStatus').textContent = '';
    document.getElementById('csvPreview').innerHTML = '';
    if (rows.length < 2) {
      document.getElementById('csvStatus').textContent = 'No data rows found — check the file.';
      return;
    }
    importCsvRows = rows;
    renderCSVPreview();
  };
  reader.readAsText(file);
}

function renderCSVPreview() {
  const rows = importCsvRows; if (!rows.length) return;
  const headers  = rows[0];
  const dataRows = rows.slice(1).filter(r => r.some(c => c));
  const cols = detectColumns(headers);
  document.getElementById('csvStatus').textContent =
    `${dataRows.length} row${dataRows.length!==1?'s':''} detected`;

  const colKeys   = ['name','affil','connTo','photo'];
  const colLabels = ['Name','Affiliation','Connects To','Photo'];
  const show = dataRows.slice(0, 8);

  document.getElementById('csvPreview').innerHTML =
    `<table class="csv-preview-tbl"><thead><tr>${
      colLabels.map((l,i) => {
        const ci = cols[colKeys[i]];
        const miss = ci < 0;
        return `<th class="${miss?'col-missing':''}">${l}${miss?' (—)':''}</th>`;
      }).join('')
    }</tr></thead><tbody>${
      show.map(r => `<tr>${
        colKeys.map(k => `<td>${esc(cols[k]>=0 ? r[cols[k]]||'' : '')}</td>`).join('')
      }</tr>`).join('')
    }</tbody></table>${dataRows.length>8?`<div class="csv-more">…and ${dataRows.length-8} more rows</div>`:''}`;
}

function layoutNewPeople(entries) {
  const HGAP = 200;
  const VGAP = 170;
  const existingPeople = pageState().people;

  const nameToNewIdx = {};
  entries.forEach((e, i) => { if (e.name) nameToNewIdx[e.name.toLowerCase()] = i; });

  const parentNewIdx   = new Array(entries.length).fill(-1);
  const parentExisting = new Array(entries.length).fill(null);
  const children       = entries.map(() => []);

  entries.forEach((e, i) => {
    if (!e.connTo) return;
    const target = e.connTo.toLowerCase();
    const j = nameToNewIdx[target];
    if (j !== undefined && j !== i) {
      parentNewIdx[i] = j;
      children[j].push(i);
    } else {
      const ep = existingPeople.find(p => p.name.toLowerCase() === target);
      if (ep) parentExisting[i] = ep;
    }
  });

  const batchRoots = entries.map((_, i) => i).filter(i => parentNewIdx[i] === -1);

  const byExisting = {};
  const freeRoots  = [];
  batchRoots.forEach(i => {
    const ep = parentExisting[i];
    if (ep) {
      if (!byExisting[ep.id]) byExisting[ep.id] = { person: ep, nodes: [] };
      byExisting[ep.id].nodes.push(i);
    } else {
      freeRoots.push(i);
    }
  });

  const leafCount = new Array(entries.length);
  function countLeaves(i) {
    if (!children[i].length) { leafCount[i] = 1; return; }
    children[i].forEach(c => countLeaves(c));
    leafCount[i] = children[i].reduce((s, c) => s + leafCount[c], 0);
  }
  entries.forEach((_, i) => { if (parentNewIdx[i] === -1) countLeaves(i); });

  const positions = new Array(entries.length);

  function place(i, centerX, y) {
    positions[i] = { x: Math.round(centerX), y: Math.round(y) };
    if (!children[i].length) return;
    let x = centerX - (leafCount[i] - 1) * HGAP / 2;
    children[i].forEach(c => {
      place(c, x + (leafCount[c] - 1) * HGAP / 2, y + VGAP);
      x += leafCount[c] * HGAP;
    });
  }

  Object.values(byExisting).forEach(({ person, nodes }) => {
    const totalLeaves = nodes.reduce((s, n) => s + leafCount[n], 0);
    let x = person.x - (totalLeaves - 1) * HGAP / 2;
    nodes.forEach(i => {
      place(i, x + (leafCount[i] - 1) * HGAP / 2, person.y + VGAP);
      x += leafCount[i] * HGAP;
    });
  });

  if (freeRoots.length) {
    const wr  = document.getElementById('wrapper');
    const cx  = (wr.clientWidth  / 2 - panX) / zoom;
    const topY = (wr.clientHeight / 5 - panY) / zoom;
    const totalLeaves = freeRoots.reduce((s, r) => s + leafCount[r], 0);
    let rx = cx - (totalLeaves - 1) * HGAP / 2;
    freeRoots.forEach(r => {
      place(r, rx + (leafCount[r] - 1) * HGAP / 2, topY);
      rx += leafCount[r] * HGAP;
    });
  }

  return { positions };
}

function submitImport() {
  let entries = [];

  if (importActiveTab === 'manual') {
    document.querySelectorAll('#importRowsContainer .imp-row').forEach(row => {
      const cells = row.querySelectorAll('.imp-cell');
      const name = cells[0].value.trim();
      if (!name) return;
      entries.push({
        name,
        affil:  cells[1].value.trim(),
        connTo: cells[2].value.trim(),
        photo:  cells[3].value.trim()
      });
    });
  } else {
    if (!importCsvRows.length) { alert('Please upload a CSV file first.'); return; }
    const headers  = importCsvRows[0];
    const dataRows = importCsvRows.slice(1).filter(r => r.some(c=>c));
    const cols = detectColumns(headers);
    if (cols.name < 0) {
      alert('Can\'t find a Name column.\nMake sure one column header contains "Name" or "Person".');
      return;
    }
    entries = dataRows
      .filter(r => (r[cols.name]||'').trim())
      .map(r => ({
        name:   (r[cols.name]||'').trim(),
        affil:  cols.affil  >= 0 ? (r[cols.affil]||'').trim()  : '',
        photo:  cols.photo  >= 0 ? (r[cols.photo]||'').trim()  : '',
        connTo: cols.connTo >= 0 ? (r[cols.connTo]||'').trim() : ''
      }));
  }

  if (!entries.length) { alert('No valid entries to import.'); return; }

  const pg = currentPage(); if (!pg) return;
  const { positions } = layoutNewPeople(entries);

  const newPeople = entries.map((e, i) => ({
    id: uid(), name: e.name, role: e.affil, photo: e.photo,
    description: '', size: 1,
    x: positions[i].x, y: positions[i].y
  }));

  pg.people.push(...newPeople);

  entries.forEach((e, i) => {
    if (!e.connTo) return;
    const from = newPeople[i];
    const target = e.connTo.toLowerCase();
    const to = pg.people.find(p => p.name.toLowerCase() === target);
    if (!to || to.id === from.id) return;
    const already = pg.conns.find(c =>
      (c.from === from.id && c.to === to.id) ||
      (c.from === to.id   && c.to === from.id)
    );
    if (!already) pg.conns.push({ id: uid(), from: from.id, to: to.id, label: '' });
  });

  save(); render(); closeImportModal();
}

// ══════════════════════════════════════════════════════
// DISCOVER — pick a person already on this board page and expand their real
// public network (live web search, 2 hops deep by default) into a full-screen
// "<name>'s Connections" view, styled like My Connections. This is the actual
// discovery feature; Route Finder's Starting Person / Target are tagged
// directly on a node's detail card (📍/🎯) instead of through this modal.
// ══════════════════════════════════════════════════════
let _dvSelectedId = null;
let _dvRunSeq = 0;

function openDiscoverModal() {
  _dvRunSeq++;
  closeDetail();
  _dvSelectedId = null;
  const st = document.getElementById('dvStatus'); st.textContent=''; st.className='dvm-status';
  document.getElementById('dvSubmitBtn').disabled = false;
  document.getElementById('dvSubmitBtn').textContent = 'RUN DISCOVER ▸';
  document.getElementById('dvProgress')?.classList.remove('on');
  const dvFill = document.getElementById('dvProgressFill'); if (dvFill) dvFill.style.width = '0%';
  renderDiscoverPicker();
  document.getElementById('discoverScrim').classList.add('open');
}
function closeDiscoverModal(cancelRun = true){
  if (cancelRun) _dvRunSeq++;
  document.getElementById('discoverScrim').classList.remove('open');
}
function discoverScrimClick(e){ if(e.target===document.getElementById('discoverScrim')) closeDiscoverModal(); }

function renderDiscoverPicker() {
  const pg = currentPage();
  const people = (pg && pg.people) || [];
  const listEl = document.getElementById('dvPersonList');
  if (!people.length) {
    listEl.innerHTML = '<div class="cp-empty">No one on this page yet.<br>Add a person to the board first.</div>';
    return;
  }
  listEl.innerHTML = people.map(p => {
    const ini2 = initials(p.name);
    const avatar = p.photo
      ? `<div class="cp-avatar"><img src="${esc(p.photo)}" onerror="this.parentNode.textContent='${ini2}'"></div>`
      : `<div class="cp-avatar">${ini2}</div>`;
    const sel = p.id === _dvSelectedId;
    return `<div class="cp-item dv-pick-item${sel?' sel':''}" onclick="selectDiscoverPerson('${p.id}')">
      ${avatar}
      <div class="cp-info">
        <div class="cp-name">${esc(p.name)}</div>
        ${p.role ? `<div class="cp-role">${esc(p.role)}</div>` : ''}
      </div>
      <span class="dv-pick-mark">${sel?'✓':''}</span>
    </div>`;
  }).join('');
}

function selectDiscoverPerson(id) {
  _dvSelectedId = id;
  renderDiscoverPicker();
}

// Polls GET /jobs/{id} (a background /discover or /connect run) until it
// finishes, calling onTick with each raw job payload {status,pct,message,...}
// so callers can drive a real progress-bar fill instead of guessing. Resolves
// with the job's result on success; throws on error/failure.
async function pollJob(jobId, onTick, intervalMs = 700, shouldStop = null) {
  while (true) {
    if (shouldStop && shouldStop()) {
      const err = new Error('Job cancelled');
      err.cancelled = true;
      throw err;
    }
    const res = await fetch(`/jobs/${jobId}`);
    if (!res.ok) {
      let detail = 'Job not found';
      try { detail = (await res.json()).detail || detail; } catch {}
      throw new Error(detail);
    }
    const job = await res.json();
    if (onTick) onTick(job);
    if (job.status === 'done') return job.result;
    if (job.status === 'cancelled' || job.status === 'cancelling') {
      const err = new Error('Job cancelled');
      err.cancelled = true;
      throw err;
    }
    if (job.status === 'error') throw new Error(job.error || 'Job failed');
    await new Promise(r => setTimeout(r, intervalMs));
  }
}

function progressTracker(fillEl) {
  let lastPct = 0;
  return job => {
    const rawPct = Number.isFinite(Number(job?.pct)) ? Number(job.pct) : 0;
    lastPct = Math.max(lastPct, Math.max(0, Math.min(100, rawPct)));
    if (fillEl) fillEl.style.width = `${lastPct}%`;
    return lastPct;
  };
}

async function submitDiscover() {
  const runSeq = ++_dvRunSeq;
  const statusEl = document.getElementById('dvStatus');
  const pg = currentPage();
  const person = _dvSelectedId && pg && pg.people.find(p => p.id === _dvSelectedId);
  if (!person) { statusEl.textContent='Select who to run Discover on.'; statusEl.className='dvm-status err'; return; }

  const depth = parseInt(document.getElementById('dvDepth').value, 10) || 2;
  const btn = document.getElementById('dvSubmitBtn');
  btn.disabled = true; btn.textContent = 'SEARCHING…';
  statusEl.textContent = `Expanding ${person.name}'s network (this reaches out to the live web, it can take a bit)…`;
  statusEl.className = 'dvm-status';
  const progressEl = document.getElementById('dvProgress');
  const fillEl = document.getElementById('dvProgressFill');
  const setProgress = progressTracker(fillEl);
  progressEl?.classList.add('on');
  if (fillEl) fillEl.style.width = '0%';

  try {
    const started = await (await fetch('/discover', {
      method: 'POST', headers: API_HEADERS,
      body: JSON.stringify({ person_name: person.name, depth }),
    })).json();
    if (started.detail) throw new Error(started.detail);
    const data = await pollJob(started.job_id, job => {
      if (runSeq !== _dvRunSeq) return;
      const pct = setProgress(job);
      if (job.message) statusEl.textContent = `[${pct}%] ${job.message}`;
    }, 700, () => runSeq !== _dvRunSeq);
    if (runSeq !== _dvRunSeq) return;
    if (!data.found) throw new Error(data.reason || `Nobody found connected to ${person.name}.`);
    closeDiscoverModal(false);
    showDiscoverResultsView(data, depth);
  } catch (e) {
    if (runSeq !== _dvRunSeq) return;
    statusEl.textContent = e.message || 'Discover failed.';
    statusEl.className = 'dvm-status err';
  } finally {
    if (runSeq === _dvRunSeq) {
      btn.disabled = false; btn.textContent = 'RUN DISCOVER ▸';
      progressEl?.classList.remove('on');
    }
  }
}

// find-or-create a plain person node by name, positioned at (x,y) if new
function findOrCreatePerson(pg, name, x, y) {
  const norm = name.toLowerCase().trim();
  let p = pg.people.find(p => p.name.toLowerCase().trim() === norm);
  if (!p) {
    p = { id: uid(), name, role: '', photo: '', description: '', size: 1, x: Math.round(x), y: Math.round(y) };
    pg.people.push(p);
  }
  return p;
}

// ══════════════════════════════════════════════════════
// DISCOVER RESULTS — "<name>'s Connections" full-screen view
// ══════════════════════════════════════════════════════
let _drPerson = '';
let _drDepth = 2;
let _drConnections = [];

function humanizeRelType(t) {
  if (!t || t === 'unknown') return 'Connection';
  return t.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function showDiscoverResultsView(data, depth) {
  _drPerson = data.person;
  _drDepth = depth;
  _drConnections = data.connections || [];

  document.getElementById('homeView').style.display = 'none';
  document.getElementById('boardView').style.display = 'none';
  document.getElementById('contactsView').style.display = 'none';
  document.getElementById('discoverResultsView').style.display = 'flex';

  document.getElementById('drSub').textContent = `// ${_drPerson.toUpperCase()}'S NETWORK`;
  document.getElementById('drTitle').textContent = `${_drPerson.toUpperCase()}'S CONNECTIONS`;
  document.getElementById('drSearch').value = '';
  setOperatorName(operatorName());
  renderDiscoverResults();
}

function closeDiscoverResultsView() {
  document.getElementById('discoverResultsView').style.display = 'none';
  document.getElementById('boardView').style.display = 'flex';
}

function renderDiscoverResults() {
  const filter = (document.getElementById('drSearch')?.value || '').toLowerCase();
  let shown = _drConnections.slice();
  if (filter) shown = shown.filter(c =>
    c.label.toLowerCase().includes(filter) ||
    humanizeRelType(c.relationship_type).toLowerCase().includes(filter)
  );

  const el = id => document.getElementById(id);
  if (el('drCount')) el('drCount').textContent = `[ ${_drConnections.length} ]`;
  if (el('drFooterR')) el('drFooterR').textContent =
    `${_drConnections.length} CONNECTION${_drConnections.length!==1?'S':''} · ${_drDepth} HOP${_drDepth!==1?'S':''} DEEP`;

  const grid = el('drGrid');
  if (!grid) return;

  if (!_drConnections.length) {
    grid.innerHTML = `<div class="hv-no-match" style="padding:60px 0">
      <div style="font-family:var(--display-hv,'Rajdhani');font-size:36px;color:var(--ink-faint);margin-bottom:10px">NOTHING FOUND</div>
      No public connections turned up for ${esc(_drPerson)} within ${_drDepth} hop${_drDepth!==1?'s':''}.
    </div>`;
    return;
  }
  if (!shown.length) {
    grid.innerHTML = `<div class="hv-no-match">// NO CONNECTIONS MATCH QUERY</div>`;
    return;
  }

  grid.innerHTML = shown.map((c, i) => {
    const ini2 = initials(c.label);
    const av = ini2;
    const pct = Math.round((c.confidence || 0) * 100);
    const openSrc = c.source_url ? `data-src-url="${esc(c.source_url)}" style="cursor:pointer"` : '';
    return `<div class="ct-card fr" ${openSrc} style="animation-delay:${(0.04+i*0.05).toFixed(2)}s">
      <span class="br tl"></span><span class="br tr"></span><span class="br bl"></span><span class="br br2"></span>
      <div class="ct-card-head">
        <div class="ct-av">${av}</div>
        <div class="ct-info">
          <div class="ct-name">${esc(c.label)}</div>
          <div class="ct-role-co">${esc(humanizeRelType(c.relationship_type))}</div>
          <div class="ct-connected">◎ ${c.hops} HOP${c.hops!==1?'S':''} OUT</div>
        </div>
      </div>
      <div class="ct-foot">
        <span class="ct-knows">CONFIDENCE <b>${pct}%</b></span>
        <span class="ct-enter">${c.source_url ? 'SOURCE ↗' : ''}</span>
      </div>
    </div>`;
  }).join('');
}

// ══════════════════════════════════════════════════════
// EXPORT / IMPORT (boards as .json) — full data always
// pulled fresh from the backend so nothing is missed.
// ══════════════════════════════════════════════════════
async function exportDashboard() {
  if (!db.boards.length) { alert('Nothing to export — create some boards first.'); return; }
  try {
    const fullBoards = await Promise.all(db.boards.map(async b => {
      const full = await (await fetch(`/boards/${b.id}`, { headers: API_HEADERS })).json();
      return {
        id: full.id, name: full.name, targetName: full.target_name, targetOrg: full.target_org,
        status: full.status, modified: new Date(full.created_at).getTime(),
        pages: (full.pages||[]).map(pg => ({ id: pg.id, name: pg.name,
          people: ((pg.elements&&pg.elements.nodes)||[]).map(backendNodeToPerson),
          conns: ((pg.elements&&pg.elements.edges)||[]).map(backendEdgeToConn) })),
      };
    }));
    const payload = { artemis_export:'v1', exported_at:new Date().toISOString(), exported_by: operatorName(), board_count: fullBoards.length, boards: fullBoards };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = `artemis-export-${new Date().toISOString().slice(0,10)}.json`; a.click();
    URL.revokeObjectURL(url);
  } catch(e) { alert('Export failed: '+e.message); }
}

function exportCurrentBoard() {
  const b = currentBoard();
  if (!b) return;
  const payload = {
    artemis_export: 'v1', exported_at: new Date().toISOString(), exported_by: operatorName(),
    board_count: 1, boards: [b],
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  const slug = (b.name || 'board').toLowerCase().replace(/[^a-z0-9]+/g,'-').slice(0,30);
  a.download = `artemis-${slug}-${new Date().toISOString().slice(0,10)}.json`;
  a.click();
  URL.revokeObjectURL(url);
}
function exportBoardCSV() {
  const b = currentBoard(); if (!b) return;
  const people = (b.pages||[]).flatMap(pg => pg.people||[]);
  const conns  = (b.pages||[]).flatMap(pg => pg.conns||[]);
  const nameOf = id => people.find(p=>p.id===id)?.name || id;
  const rows = people.map(p => {
    const myConns = conns.filter(c=>c.from===p.id||c.to===p.id).map(c=>nameOf(c.from===p.id?c.to:c.from)).join(' | ');
    return [p.name, p.role||'', p.photo||'', p.description||'', myConns];
  });
  const header = ['Name','Role','Photo','Description','Connected To'];
  const csvLines = [header, ...rows].map(row =>
    row.map(v => '"' + String(v).replace(/"/g, '""') + '"').join(',')
  );
  const blob = new Blob([csvLines.join('\n')], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url;
  const slug = (b.name||'board').toLowerCase().replace(/[^a-z0-9]+/g,'-').slice(0,30);
  a.download = `artemis-${slug}-${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function triggerImport() {
  const inp = document.getElementById('hvImportFile');
  if (inp) { inp.value = ''; inp.click(); }
}

let _importPayload = null;

function handleImportFile(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    let payload;
    try { payload = JSON.parse(ev.target.result); }
    catch { alert('// PARSE ERROR — file is not valid JSON.'); return; }

    if (!payload.artemis_export || !Array.isArray(payload.boards) || !payload.boards.length) {
      alert('// FORMAT ERROR — this does not look like an Artemis export file.');
      return;
    }

    _importPayload = payload;
    showImportPreview(payload);
  };
  reader.readAsText(file);
}

function showImportPreview(payload) {
  const scrim = document.getElementById('hvImportScrim');
  if (!scrim) return;

  const date = payload.exported_at ? new Date(payload.exported_at).toLocaleDateString() : '–';
  const by   = payload.exported_by || '–';
  document.getElementById('hvImportMeta').innerHTML =
    `<b>${payload.boards.length}</b> board${payload.boards.length!==1?'s':''} · exported by <b>${esc(by)}</b> on <b>${date}</b>`;

  document.getElementById('hvImportList').innerHTML = payload.boards.map(b => {
    const nodesCt = (b.pages||[]).reduce((n,p)=>n+(p.people?.length||0),0);
    const tgt = b.targetName || getTargetPerson(b.pages&&b.pages[0]).name;
    const status = b.status || 'active';
    return `<div class="hv-import-row">
      <span class="ir-pill">${status.toUpperCase()}</span>
      <span class="ir-name">${esc(b.name||'Untitled')}</span>
      <span class="ir-tgt">${esc(tgt)}</span>
      <span class="ir-nodes">${nodesCt} node${nodesCt!==1?'s':''}</span>
    </div>`;
  }).join('');

  document.getElementById('hvImportConfirm').textContent =
    `IMPORT ${payload.boards.length} BOARD${payload.boards.length!==1?'S':''} ▸`;

  scrim.classList.add('open');
}

function closeImportPreview() {
  const scrim = document.getElementById('hvImportScrim');
  if (scrim) scrim.classList.remove('open');
  _importPayload = null;
}

function hvImportScrimClick(e) {
  if (e.target === document.getElementById('hvImportScrim')) closeImportPreview();
}

async function confirmImport() {
  if (!_importPayload || !Array.isArray(_importPayload.boards)) return;
  const btn = document.getElementById('hvImportConfirm');
  btn.disabled = true; const origText = btn.textContent; btn.textContent = 'IMPORTING…';
  try {
    for (const b of _importPayload.boards) {
      const created = await (await fetch('/boards', { method:'POST', headers:API_HEADERS,
        body: JSON.stringify({ name: b.name||'Untitled', target_name: b.targetName||'', target_org: b.targetOrg||'' }) })).json();
      const pages = b.pages && b.pages.length ? b.pages : [{ name:'Page 1', people:[], conns:[] }];
      const detail = await (await fetch(`/boards/${created.id}`, { headers: API_HEADERS })).json();
      for (let i = 0; i < pages.length; i++) {
        const srcPage = pages[i];
        let pageId;
        if (i === 0) { pageId = detail.pages[0].id; }
        else {
          const np = await (await fetch(`/boards/${created.id}/pages`, { method:'POST', headers:API_HEADERS,
            body: JSON.stringify({ name: srcPage.name||('Page '+(i+1)) }) })).json();
          pageId = np.id;
        }
        const elements = { nodes: (srcPage.people||[]).map(personToBackendNode), edges: (srcPage.conns||[]).map(connToBackendEdge) };
        await fetch(`/boards/${created.id}/pages/${pageId}`, { method:'PATCH', headers:API_HEADERS,
          body: JSON.stringify({ name: srcPage.name, elements }) });
      }
    }
    closeImportPreview();
    await loadBoardsFromBackend();
    renderHome();
  } catch(e) {
    alert('Import failed partway through: '+e.message);
  } finally { btn.disabled = false; btn.textContent = origText; }
}

// ══════════════════════════════════════════════════════
// MY CONNECTIONS — backed by /network/profiles. Contact-to-
// contact "Knows" links and photos have no backend field, so
// they're layered on top via localStorage (flagged in the UI).
// ══════════════════════════════════════════════════════
let selectedContactId = null;
let _linkFromId = null;
let _linkSelectedId = null;

async function showContactsView() {
  document.getElementById('homeView').style.display = 'none';
  document.getElementById('boardView').style.display = 'none';
  document.getElementById('contactsView').style.display = 'flex';
  closeDetailRail();
  setOperatorName(operatorName());
  await loadContactsFromBackend();
  renderContacts();
}

function renderContacts() {
  const filter = (document.getElementById('ctSearch')?.value || '').toLowerCase();
  const contacts = db.contacts || [];

  const totalLinks = contacts.reduce((n,c)=>(n+(c.conns||[]).length),0);
  const el = id => document.getElementById(id);
  if(el('ctCount')) el('ctCount').textContent = `[ ${contacts.length} ]`;
  if(el('ctFooterR')) el('ctFooterR').textContent = `${contacts.length} CONTACT${contacts.length!==1?'S':''} · ${totalLinks} LINK${totalLinks!==1?'S':''}`;

  const grid = el('ctGrid');
  if (!grid) return;

  let shown = contacts.slice();
  if (filter) shown = shown.filter(c =>
    c.name.toLowerCase().includes(filter) ||
    (c.role||'').toLowerCase().includes(filter) ||
    (c.company||'').toLowerCase().includes(filter)
  );

  if (!contacts.length) {
    grid.innerHTML = `<div class="hv-no-match" style="padding:60px 0">
      <div style="font-family:var(--display-hv,'Rajdhani');font-size:36px;color:var(--ink-faint);margin-bottom:10px">NO CONTACTS</div>
      Click <span style="color:var(--accent)">+ ADD CONTACT</span> to start building your network.
    </div>`;
    return;
  }
  if (!shown.length) {
    grid.innerHTML = `<div class="hv-no-match">// NO CONTACTS MATCH QUERY</div>`;
    return;
  }

  const adj = buildAdj();
  grid.innerHTML = shown.map((c,i) => {
    const knows = (adj[c.id]||[]).length;
    const ini2  = initials(c.name);
    const av    = c.photo
      ? `<img src="${esc(c.photo)}" onerror="this.style.display='none';this.nextElementSibling.style.display='grid'" alt="">`
        + `<span style="display:none">${esc(ini2)}</span>`
      : esc(ini2);
    const roleStr = [c.role, c.company].filter(Boolean).join(' · ');
    return `<div class="ct-card fr" data-id="${c.id}" onclick="selectContact('${c.id}')" style="animation-delay:${(0.04+i*0.05).toFixed(2)}s">
      <span class="br tl"></span><span class="br tr"></span><span class="br bl"></span><span class="br br2"></span>
      <div class="ct-card-head">
        <div class="ct-av">${av}</div>
        <div class="ct-info">
          <div class="ct-name">${esc(c.name)}</div>
          <div class="ct-role-co">${esc(roleStr||'–')}</div>
          ${c.email ? `<div class="ct-email">✉ ${esc(c.email)}</div>` : ''}
          ${c.connectedOn ? `<div class="ct-connected">◎ CONNECTED ${esc(c.connectedOn)}</div>` : ''}
        </div>
      </div>
      ${c.description ? `<div class="ct-desc">${esc(c.description)}</div>` : ''}
      <div class="ct-foot">
        <span class="ct-knows">KNOWS <b>${knows}</b></span>
        <span class="ct-enter">VIEW ▸</span>
      </div>
    </div>`;
  }).join('') + `<div class="ct-ghost fr" onclick="showAddContactModal()">
    <span class="br tl"></span><span class="br tr"></span><span class="br bl"></span><span class="br br2"></span>
    <div class="ct-ghost-in">
      <div class="ct-ghost-ring">+</div>
      <div class="ct-ghost-lbl">ADD CONTACT</div>
    </div>
  </div>`;
}

function buildAdj() {
  const adj = {};
  (db.contacts||[]).forEach(c => { if(!adj[c.id]) adj[c.id]=[]; });
  (db.contacts||[]).forEach(c => {
    (c.conns||[]).forEach(otherId => {
      if(!adj[c.id]) adj[c.id]=[];
      if(!adj[otherId]) adj[otherId]=[];
      if(!adj[c.id].includes(otherId))    adj[c.id].push(otherId);
      if(!adj[otherId].includes(c.id))    adj[otherId].push(c.id);
    });
  });
  return adj;
}

function selectContact(id) {
  selectedContactId = id;
  document.querySelectorAll('#ctGrid .ct-card').forEach(el =>
    el.classList.toggle('sel', el.dataset.id === id)
  );
  showContactRail(id);
}

function showContactRail(id) {
  const c = (db.contacts||[]).find(x=>x.id===id);
  if (!c) return;
  const adj = buildAdj();
  const knownIds = adj[c.id] || [];
  const ini2 = initials(c.name);

  const avEl = document.getElementById('ctrAvatar');
  if (avEl) {
    if (c.photo) {
      avEl.innerHTML = `<img src="${esc(c.photo)}" onerror="this.style.display='none'" alt=""><span style="display:none">${esc(ini2)}</span>`;
    } else {
      avEl.textContent = ini2;
    }
  }
  const nameEl = document.getElementById('ctrName'); if(nameEl) nameEl.textContent = c.name;
  const roleEl = document.getElementById('ctrRole');
  if(roleEl) roleEl.textContent = [c.role,c.company].filter(Boolean).join(' · ') || '–';
  const descEl = document.getElementById('ctrDesc');
  if(descEl) { descEl.textContent = c.description || ''; descEl.style.display = c.description ? '' : 'none'; }
  const codeEl = document.getElementById('ctrCode');
  if(codeEl) codeEl.textContent = `// CONTACT`;
  const emailRow = document.getElementById('ctrEmailRow');
  const dateRow  = document.getElementById('ctrDateRow');
  const metaBox  = document.getElementById('ctrMeta');
  if(emailRow) { emailRow.style.display = c.email ? '' : 'none'; const el=document.getElementById('ctrEmail'); if(el) el.textContent = c.email||''; }
  if(dateRow)  { dateRow.style.display  = c.connectedOn ? '' : 'none'; const el=document.getElementById('ctrDate');  if(el) el.textContent = c.connectedOn||''; }
  if(metaBox)  { metaBox.style.display  = (c.email || c.connectedOn) ? '' : 'none'; }

  const knowsEl = document.getElementById('ctrKnows');
  if (knowsEl) {
    if (!knownIds.length) {
      knowsEl.innerHTML = `<div class="ctr-empty">No links yet — click below to connect this person.</div>`;
    } else {
      knowsEl.innerHTML = knownIds.map(kid => {
        const other = (db.contacts||[]).find(x=>x.id===kid);
        if (!other) return '';
        const r = [other.role,other.company].filter(Boolean).join(' · ');
        return `<div class="ctr-knows-item">
          <div>
            <div class="kn">${esc(other.name)}</div>
            ${r?`<div class="kr">${esc(r)}</div>`:''}
          </div>
          <button class="rm-link" onclick="unlinkContacts('${c.id}','${kid}')" title="Remove link">✕</button>
        </div>`;
      }).join('');
    }
  }

  document.getElementById('ctRail').classList.add('open');
  document.getElementById('ctRailScrim').classList.add('open');
}

function closeContactRail() {
  document.getElementById('ctRail')?.classList.remove('open');
  document.getElementById('ctRailScrim')?.classList.remove('open');
  document.querySelectorAll('#ctGrid .ct-card').forEach(el => el.classList.remove('sel'));
  selectedContactId = null;
}

function showAddContactModal() {
  ['camName','camRole','camCompany','camEmail','camPhoto','camDesc'].forEach(id => {
    const el = document.getElementById(id); if(el) el.value='';
  });
  document.getElementById('ctAddScrim')?.classList.add('open');
  setTimeout(() => document.getElementById('camName')?.focus(), 60);
}
function closeAddContactModal() {
  document.getElementById('ctAddScrim')?.classList.remove('open');
}
function ctAddScrimClick(e) {
  if (e.target === document.getElementById('ctAddScrim')) closeAddContactModal();
}
async function submitAddContact() {
  const name = (document.getElementById('camName')?.value||'').trim();
  if (!name) { document.getElementById('camName')?.focus(); return; }
  const photo = (document.getElementById('camPhoto')?.value||'').trim();
  try {
    const res = await fetch('/network/profiles', { method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        name,
        title:   (document.getElementById('camRole')?.value||'').trim(),
        company: (document.getElementById('camCompany')?.value||'').trim(),
        email:   (document.getElementById('camEmail')?.value||'').trim(),
        notes:   (document.getElementById('camDesc')?.value||'').trim(),
      }) });
    if (!res.ok) throw new Error(await res.text());
    const created = await res.json();
    if (photo) { const photos=_localPhotos(); photos[created.id]=photo; _saveLocalPhotos(photos); }
    closeAddContactModal();
    await loadContactsFromBackend();
    renderContacts();
  } catch(e) { alert('Failed to add contact: '+e.message); }
}

function showLinkContactModal() {
  if (!selectedContactId) return;
  _linkFromId = selectedContactId;
  _linkSelectedId = null;
  const c = (db.contacts||[]).find(x=>x.id===_linkFromId);
  const whoEl = document.getElementById('clmWho'); if(whoEl) whoEl.textContent = c?.name || 'this person';
  const searchEl = document.getElementById('clmSearch'); if(searchEl) searchEl.value = '';
  document.getElementById('ctLinkScrim')?.classList.add('open');
  renderLinkList();
  setTimeout(() => document.getElementById('clmSearch')?.focus(), 60);
}
function closeLinkContactModal() {
  document.getElementById('ctLinkScrim')?.classList.remove('open');
  _linkFromId = null; _linkSelectedId = null;
}
function ctLinkScrimClick(e) {
  if (e.target === document.getElementById('ctLinkScrim')) closeLinkContactModal();
}
function renderLinkList() {
  const filter = (document.getElementById('clmSearch')?.value||'').toLowerCase();
  const adj = buildAdj();
  const alreadyLinked = adj[_linkFromId] || [];
  let candidates = (db.contacts||[]).filter(c =>
    c.id !== _linkFromId && !alreadyLinked.includes(c.id)
  );
  if (filter) candidates = candidates.filter(c =>
    c.name.toLowerCase().includes(filter) ||
    (c.role||'').toLowerCase().includes(filter) ||
    (c.company||'').toLowerCase().includes(filter)
  );
  const list = document.getElementById('clmList');
  if (!list) return;
  if (!candidates.length) {
    list.innerHTML = `<div class="clm-empty">${filter ? '// No matches' : '// All contacts already linked'}</div>`;
    return;
  }
  list.innerHTML = candidates.map(c => {
    const ini2 = initials(c.name);
    const avHtml = c.photo
      ? `<img src="${esc(c.photo)}" onerror="this.style.display='none'" alt="">`
      : ini2;
    const r = [c.role,c.company].filter(Boolean).join(' · ');
    return `<div class="clm-row${_linkSelectedId===c.id?' selected':''}" onclick="selectLinkTarget('${c.id}')">
      <div class="clm-av">${avHtml}</div>
      <div class="clm-name">${esc(c.name)}</div>
      ${r?`<div class="clm-role">${esc(r)}</div>`:''}
    </div>`;
  }).join('');
}
function selectLinkTarget(id) {
  _linkSelectedId = id;
  renderLinkList();
}
function submitLink() {
  if (!_linkFromId || !_linkSelectedId) return;
  const links = _localLinks();
  links[_linkFromId] = links[_linkFromId] || [];
  if (!links[_linkFromId].includes(_linkSelectedId)) links[_linkFromId].push(_linkSelectedId);
  _saveLocalLinks(links);
  const from = (db.contacts||[]).find(x=>x.id===_linkFromId);
  if (from) { from.conns = from.conns || []; if (!from.conns.includes(_linkSelectedId)) from.conns.push(_linkSelectedId); }
  closeLinkContactModal();
  showContactRail(_linkFromId);
  renderContacts();
}
function unlinkContacts(fromId, toId) {
  const links = _localLinks();
  links[fromId] = (links[fromId]||[]).filter(id=>id!==toId);
  links[toId]   = (links[toId]||[]).filter(id=>id!==fromId);
  _saveLocalLinks(links);
  const from = (db.contacts||[]).find(x=>x.id===fromId);
  const to   = (db.contacts||[]).find(x=>x.id===toId);
  if (from) from.conns = (from.conns||[]).filter(id=>id!==toId);
  if (to)   to.conns   = (to.conns||[]).filter(id=>id!==fromId);
  showContactRail(fromId);
  renderContacts();
}
async function clearAllContacts() {
  const n = (db.contacts||[]).length;
  if (!n) return;
  if (!confirm(`Delete all ${n} contacts? This cannot be undone.`)) return;
  try {
    await fetch('/network/profiles', { method: 'DELETE' });
    localStorage.removeItem('artemis_contact_links');
    localStorage.removeItem('artemis_contact_photos');
    db.contacts = [];
    closeContactRail();
    renderContacts();
  } catch(e) { alert('Failed to clear contacts: '+e.message); }
}

function deleteCurrentContact() {
  if (!selectedContactId) return;
  alert("Removing a single contact isn't supported by the server yet — it would just reappear next time this page loads. Use \"Clear All\" if you want to reset your whole contact list.");
}

// ══════════════════════════════════════════════════════
// ROUTE FINDING — real backend calls.
//  - Contacts page: match uploaded contacts against the public
//    graph (/match + /candidate-paths).
//  - Board page: live public-web path between a board person
//    and any target (/connect).
// ══════════════════════════════════════════════════════
function renderRoutePath(path) {
  if (!path || !path.length) return `<div class="rt-no-path">// No path found.</div>`;
  return `<div class="rt-path">${path.map(step => {
    const label = step.role || step.company
      ? `${step.role}${step.role && step.company?' · ':''}${step.company}`
      : '';
    const sym = step.kind === 'you' ? '●' : step.kind === 'target' ? '◆' : '○';
    return `<div class="rt-pnode k-${step.kind}">
      <div class="pip">${sym}</div>
      <div class="pinfo">
        <div class="n">${esc(step.name)}</div>
        ${label?`<div class="r">${esc(label)}</div>`:''}
      </div>
    </div>`;
  }).join('')}</div>`;
}

function candidatePathToSteps(cp) {
  const nodes = (cp.path && cp.path.path) || [];
  return nodes.map((n,i) => ({
    name: n.label,
    role: n.reason || '',
    company: '',
    kind: n.node_type === 'you' ? 'you' : (i === nodes.length - 1 ? 'target' : 'node'),
  }));
}

async function execContactRoute() {
  const target = document.getElementById('rtTarget')?.value||'';
  const resultEl = document.getElementById('rtResult');
  if (!resultEl) return;
  if (!target.trim()) { resultEl.innerHTML = ''; return; }
  const progressEl = document.getElementById('rtProgress');
  resultEl.innerHTML = `<div class="rt-no-path">// Searching public graph…</div>`;
  progressEl?.classList.add('on');
  try {
    const people = await (await fetch('/people')).json();
    const norm = target.trim().toLowerCase();
    const person = people.find(p=>p.canonical_name.toLowerCase().trim()===norm)
      || people.find(p=>p.canonical_name.toLowerCase().includes(norm));
    if (!person) {
      resultEl.innerHTML = `<div class="rt-no-path">// "${esc(target)}" isn't in the graph yet — add them to a board with <b style="color:var(--accent)">⬆ Import</b> or Manual Connections first.</div>`;
      return;
    }
    const matchRes = await (await fetch(`/match/${person.id}`, { method:'POST' })).json();
    const paths = await (await fetch('/candidate-paths')).json();
    const relevant = paths.filter(p=>p.target_person_id===person.id).sort((a,b)=>b.score-a.score);
    if (!relevant.length) {
      resultEl.innerHTML = `<div class="rt-no-path">// ${matchRes.matches} contact match(es), but no candidate path found to "${esc(person.canonical_name)}" yet.</div>`;
      return;
    }
    resultEl.innerHTML = renderRoutePath(candidatePathToSteps(relevant[0]));
  } catch(e) {
    resultEl.innerHTML = `<div class="rt-no-path">// Error: ${esc(e.message)}</div>`;
  } finally {
    progressEl?.classList.remove('on');
  }
}

function _routeEndpoints() {
  const pg = currentPage();
  const start = pg && pg.startPersonId && pg.people.find(p=>p.id===pg.startPersonId);
  const target = pg && pg.targetPersonId && pg.people.find(p=>p.id===pg.targetPersonId);
  return { pg, start, target };
}

function showBoardRouteFinder() {
  const { start, target } = _routeEndpoints();
  const startEl = document.getElementById('bvrStartDisplay');
  const targetEl = document.getElementById('bvrTargetDisplay');
  startEl.textContent = start ? start.name : '— tag a node 📍 in its detail card —';
  startEl.classList.toggle('set', !!start);
  targetEl.textContent = target ? target.name : '— tag a node 🎯 in its detail card —';
  targetEl.classList.toggle('set', !!target);
  if (!_bvrActiveJobId) document.getElementById('bvrRunBtn').disabled = !(start && target);
  document.getElementById('bvrResult').innerHTML = '';
  document.getElementById('bvrResultLbl').style.display = 'none';
  document.getElementById('bvrProgress')?.classList.remove('on');
  const bvrFill = document.getElementById('bvrProgressFill'); if (bvrFill) bvrFill.style.width = '0%';
  // Cleared on every open, not carried over: stale context from a PREVIOUS
  // pair's disambiguation ("biotech founder") would silently misdirect a
  // search for a totally different, unrelated pair if left in place.
  const ctxA = document.getElementById('bvrContextA'); if (ctxA) ctxA.value = '';
  const ctxB = document.getElementById('bvrContextB'); if (ctxB) ctxB.value = '';
  document.getElementById('bvRoutePanel')?.classList.add('open');
  document.getElementById('bvRoutePanelScrim')?.classList.add('open');
  if (start && target) execBoardRoute();
}
function closeBoardRouteFinder() {
  if (_bvrActiveJobId) cancelBoardRoute(true);
  document.getElementById('bvRoutePanel')?.classList.remove('open');
  document.getElementById('bvRoutePanelScrim')?.classList.remove('open');
}

let _bvrActiveJobId = null;
let _bvrCancelRequested = false;
let _bvrRunSeq = 0;

function setBoardRouteRunning(running) {
  const runBtn = document.getElementById('bvrRunBtn');
  const cancelBtn = document.getElementById('bvrCancelBtn');
  if (runBtn) {
    runBtn.disabled = running || !(_routeEndpoints().start && _routeEndpoints().target);
    runBtn.textContent = running ? 'TRACING ROUTE…' : 'TRACE ROUTE ▸';
  }
  if (cancelBtn) {
    cancelBtn.style.display = running ? 'flex' : 'none';
    cancelBtn.disabled = false;
    cancelBtn.textContent = 'CANCEL ROUTE ✕';
  }
}

async function cancelBoardRoute(silent = false) {
  const jobId = _bvrActiveJobId;
  _bvrCancelRequested = true;
  const cancelBtn = document.getElementById('bvrCancelBtn');
  const lbl = document.getElementById('bvrResultLbl');
  const resultEl = document.getElementById('bvrResult');
  if (cancelBtn) {
    cancelBtn.disabled = true;
    cancelBtn.textContent = 'CANCELLING…';
  }
  if (!silent) {
    if (lbl) { lbl.style.display = 'block'; lbl.textContent = '// CANCELLING…'; }
    if (resultEl) resultEl.innerHTML = `<div class="bvr-no-path">Stopping this route search. In-flight web requests may take a moment to unwind.</div>`;
  }
  if (!jobId) return;
  try {
    await fetch(`/jobs/${jobId}/cancel`, { method:'POST', headers:API_HEADERS });
  } catch(e) {
    console.warn('Failed to cancel route job', e);
  }
}

async function execBoardRoute() {
  const runSeq = ++_bvrRunSeq;
  const { start, target } = _routeEndpoints();
  const resultEl = document.getElementById('bvrResult');
  const lbl      = document.getElementById('bvrResultLbl');
  if (!resultEl) return;
  if (!start || !target) {
    lbl.style.display='block'; lbl.textContent='// NO ENDPOINTS SET';
    resultEl.innerHTML = `<div class="bvr-no-path">Tag a node 📍 in its detail card to set a starting person and target for this page first.</div>`;
    return;
  }
  const depth = parseInt(document.getElementById('bvrDepth')?.value,10) || 2;
  const contextA = (document.getElementById('bvrContextA')?.value || '').trim();
  const contextB = (document.getElementById('bvrContextB')?.value || '').trim();
  const progressEl = document.getElementById('bvrProgress');
  const fillEl = document.getElementById('bvrProgressFill');
  const setProgress = progressTracker(fillEl);
  if (_bvrActiveJobId) await cancelBoardRoute(true);
  if (runSeq !== _bvrRunSeq) return;
  _bvrActiveJobId = null;
  _bvrCancelRequested = false;
  setBoardRouteRunning(true);
  lbl.style.display='block'; lbl.textContent='// SEARCHING…';
  resultEl.innerHTML = `<div class="bvr-no-path">Searching the public web for every route between "${esc(start.name)}" and "${esc(target.name)}"…</div>`;
  progressEl?.classList.add('on');
  if (fillEl) fillEl.style.width = '0%';
  try {
    const started = await (await fetch('/connect', { method:'POST', headers:API_HEADERS,
      body: JSON.stringify({ person_a: start.name, person_b: target.name, depth,
                             context_a: contextA, context_b: contextB }) })).json();
    if (started.detail) throw new Error(started.detail);
    if (runSeq !== _bvrRunSeq) return;
    _bvrActiveJobId = started.job_id;
    if (_bvrCancelRequested) {
      await cancelBoardRoute(true);
      const err = new Error('Job cancelled');
      err.cancelled = true;
      throw err;
    }
    const data = await pollJob(started.job_id, job => {
      if (runSeq !== _bvrRunSeq) return;
      const pct = setProgress(job);
      if (job.message) lbl.textContent = `// [${pct}%] ${job.message.toUpperCase()}`;
    }, 700, () => _bvrCancelRequested || runSeq !== _bvrRunSeq);
    if (runSeq !== _bvrRunSeq) return;
    if (!data.connected) {
      lbl.textContent = '// NO ROUTE';
      resultEl.innerHTML = `<div class="bvr-no-path">No public path found between "${esc(start.name)}" and "${esc(target.name)}" — ${esc(data.reason||'')}. Try a higher depth.</div>`;
      return;
    }
    const routes = (data.paths && data.paths.length) ? data.paths : [{ path: data.path, hops: data.hops, score: data.score }];
    lbl.textContent = `// ${routes.length} ROUTE${routes.length!==1?'S':''} FOUND`;
    resultEl.innerHTML = routes.map((route, i) => {
      const steps = (route.path||[]).map((n,idx,arr) => ({
        name: n.label, role: n.relationship_from_previous||'', company: '',
        kind: idx===0 ? 'you' : (idx===arr.length-1 ? 'target' : 'node'),
      }));
      return `<div style="margin-bottom:16px"><div class="bvr-flbl">ROUTE ${i+1} · ${route.hops} HOPS · SCORE ${route.score}</div>${renderRoutePath(steps)}</div>`;
    }).join('');
    mergeConnectResultIntoBoard(data);
  } catch(e) {
    if (runSeq !== _bvrRunSeq) return;
    if (e.cancelled || _bvrCancelRequested) {
      lbl.textContent = '// CANCELLED';
      resultEl.innerHTML = `<div class="bvr-no-path">Route search cancelled.</div>`;
    } else {
      lbl.textContent = '// ERROR';
      resultEl.innerHTML = `<div class="bvr-no-path">${esc(e.message)}</div>`;
    }
  } finally {
    if (runSeq === _bvrRunSeq) {
      _bvrActiveJobId = null;
      _bvrCancelRequested = false;
      setBoardRouteRunning(false);
      progressEl?.classList.remove('on');
    }
  }
}

function mergeConnectResultIntoBoard(data) {
  const pg = currentPage(); if (!pg) return;
  const routes = (data.paths && data.paths.length) ? data.paths : (data.path ? [{ path: data.path }] : []);
  const byName = new Map(pg.people.map(p => [p.name.toLowerCase(), p]));
  const w = document.getElementById('wrapper');
  const cx = (w.clientWidth/2 - panX) / zoom, cy = (w.clientHeight/2 - panY) / zoom;
  // golden-angle spacing (never lands on the 0°/180° axis where start/target
  // already sit) so new bridge nodes never stack on top of existing ones
  let i = 0;

  routes.forEach(route => {
    const path = route.path || [];
    path.forEach(hop => {
      const key = hop.label.toLowerCase();
      let p = byName.get(key);
      if (!p) {
        const angle = Math.PI/2 + (i * 137.508 * Math.PI / 180); i++;
        const radius = 200 + Math.floor(i/8)*140;
        p = { id: uid(), name: hop.label, role: '', photo: '', description: '', size: 1,
          x: Math.round(cx + Math.cos(angle)*radius), y: Math.round(cy + Math.sin(angle)*radius) };
        pg.people.push(p); byName.set(key, p);
      }
    });
    for (let idx = 1; idx < path.length; idx++) {
      const a = byName.get(path[idx-1].label.toLowerCase());
      const b = byName.get(path[idx].label.toLowerCase());
      if (!a || !b) continue;
      const already = pg.conns.find(c => (c.from===a.id&&c.to===b.id)||(c.from===b.id&&c.to===a.id));
      if (!already) pg.conns.push({ id: uid(), from: a.id, to: b.id, label: path[idx].relationship_from_previous||'' });
    }
  });
  save(); render();
}

// ══════════════════════════════════════════════════════
// MY CONNECTIONS IMPORT — two tabs, one modal:
//   • LinkedIn CSV — client-side parse for the preview list only; the import
//     uploads the raw file to /network/upload (which parses it robustly).
//   • iPhone contacts — a .vcf is parsed here so the user can pick who to bring
//     over, then the chosen cards POST to /network/contacts/import, which runs
//     the same ingestion as the CSV path (de-dupe, "You" edge, graph edges).
// Both land in My Connections, not on a board page.
// ══════════════════════════════════════════════════════
let _liFile = null;
let _netImportTab = 'csv';

function showLinkedInImport() {
  _liFile = null;
  document.getElementById('liPreview').style.display = 'none';
  document.getElementById('liImportBtn').disabled = true;
  resetVcfTab();
  switchNetImportTab('csv');
  document.getElementById('liScrim').classList.add('open');
}
function closeLinkedInImport() {
  document.getElementById('liScrim').classList.remove('open');
  _liFile = null;
}
function switchNetImportTab(tab) {
  _netImportTab = tab;
  document.getElementById('liCsvSec').style.display = tab === 'csv' ? '' : 'none';
  document.getElementById('liVcfSec').style.display = tab === 'vcf' ? '' : 'none';
  document.getElementById('liTabCsv').className = 'li-tab' + (tab === 'csv' ? ' on' : '');
  document.getElementById('liTabVcf').className = 'li-tab' + (tab === 'vcf' ? ' on' : '');
  document.getElementById('liTitle').textContent =
    tab === 'csv' ? 'LINKEDIN CONNECTIONS' : 'IPHONE CONTACTS';
  refreshNetImportBtn();
}
function refreshNetImportBtn() {
  const btn = document.getElementById('liImportBtn');
  btn.disabled = _netImportTab === 'csv' ? !_liFile : !vcfPicked().length;
}
function confirmNetImport() {
  return _netImportTab === 'csv' ? confirmLinkedInImport() : confirmVcfImport();
}
function liScrimClick(e) {
  if (e.target === document.getElementById('liScrim')) closeLinkedInImport();
}
function liDrop(e) {
  e.preventDefault();
  document.getElementById('liDrop').classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) processLinkedInFile(file);
}
function liFileChange(e) {
  const file = e.target.files[0];
  if (file) processLinkedInFile(file);
  e.target.value = '';
}
function processLinkedInFile(file) {
  _liFile = file;
  const reader = new FileReader();
  reader.onload = ev => {
    const parsed = parseLinkedInCSV(ev.target.result) || [];
    showLinkedInPreview(parsed);
  };
  reader.readAsText(file);
}
function parseLinkedInCSV(text) {
  const lines = text.split(/\r?\n/);
  let headerIdx = -1;
  let headers = [];
  for (let i = 0; i < Math.min(15, lines.length); i++) {
    if (/first.name/i.test(lines[i]) && /last.name/i.test(lines[i])) {
      headerIdx = i; headers = parseCSVLine(lines[i]); break;
    }
  }
  if (headerIdx === -1) return null;
  const contacts = [];
  for (let i = headerIdx + 1; i < lines.length; i++) {
    const line = lines[i].trim(); if (!line) continue;
    const fields = parseCSVLine(line);
    const row = {};
    headers.forEach((h, idx) => row[h.trim().toLowerCase().replace(/\s+/g,'_')] = (fields[idx]||'').trim());
    const first = row['first_name'] || row['firstname'] || '';
    const last  = row['last_name']  || row['lastname']  || '';
    const name  = `${first} ${last}`.trim();
    if (!name) continue;
    const role    = row['position'] || row['title'] || '';
    const company = row['company']  || row['organization'] || '';
    contacts.push({ name, role, company });
  }
  return contacts;
}
function parseCSVLine(line) {
  const fields = []; let cur = ''; let inQ = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') { inQ = !inQ; }
    else if (ch === ',' && !inQ) { fields.push(cur); cur = ''; }
    else { cur += ch; }
  }
  fields.push(cur);
  return fields;
}
function showLinkedInPreview(contacts) {
  const existing = new Set((db.contacts||[]).map(c => c.name.toLowerCase()));
  const newOnes  = contacts.filter(c => !existing.has(c.name.toLowerCase()));
  const dupes    = contacts.length - newOnes.length;
  const hdr = document.getElementById('liPreviewHdr');
  if (hdr) hdr.textContent = `${newOnes.length} new contacts${dupes ? ` · ${dupes} already in Artemis (server will de-dupe)` : ''}`;
  const rows = document.getElementById('liPreviewRows');
  if (rows) {
    rows.innerHTML = newOnes.slice(0,50).map(c => {
      const co = [c.role, c.company].filter(Boolean).join(' · ');
      return `<div class="li-preview-row">
        <div class="lpr-name">${esc(c.name)}</div>
        ${co ? `<div class="lpr-co">${esc(co)}</div>` : ''}
      </div>`;
    }).join('') + (newOnes.length > 50 ? `<div class="li-preview-row" style="color:var(--ink-faint,#555)">…and ${newOnes.length-50} more</div>` : '');
  }
  document.getElementById('liPreview').style.display = 'block';
  document.getElementById('liImportBtn').disabled = !contacts.length;
}
async function confirmLinkedInImport() {
  if (!_liFile) return;
  const btn = document.getElementById('liImportBtn');
  btn.disabled = true; const orig = btn.textContent; btn.textContent = 'IMPORTING…';
  try {
    const fd = new FormData(); fd.append('file', _liFile);
    const res = await fetch('/network/upload', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    closeLinkedInImport();
    await loadContactsFromBackend();
    renderContacts();
    alert(`Imported: ${data.ingested.created} new, ${data.ingested.updated} updated, ${data.ingested.skipped} skipped.`);
  } catch(e) {
    alert('Import failed: '+e.message);
  } finally { btn.disabled = false; btn.textContent = orig; }
}

// ── iPhone contacts (.vcf / vCard) ────────────────────────────────────────
// iOS exposes no web API for the address book, so the file is the interface:
// a .vcf holding one card or a whole address book. Parsing happens here so the
// user can choose who to import; only the picked cards are ever sent.
// Chromium on Android also has the Contact Picker API — offered when present.
let vcfContacts = [];      // {name, org, title, email, tel, sel}
let _vcfShownCount = 0;

function resetVcfTab() {
  vcfContacts = [];
  const fi = document.getElementById('vcfFileIn'); if (fi) fi.value = '';
  const se = document.getElementById('vcfSearch'); if (se) se.value = '';
  document.getElementById('vcfPreview').style.display = 'none';
  document.getElementById('vcfSelectBar').style.display = 'none';
  document.getElementById('vcfFixWrap').style.display = 'none';
  document.getElementById('vcfList').innerHTML = '';
  document.getElementById('vcfFixList').innerHTML = '';
  const canPick = !!(navigator.contacts && navigator.contacts.select && window.ContactsManager);
  document.getElementById('vcfPickBtn').style.display = canPick ? '' : 'none';
}

// RFC 6350 folding: a line beginning with a space/tab continues the previous one.
function unfoldVcf(text) {
  return text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').replace(/\n[ \t]/g, '');
}

function vcfUnescape(v) {
  return String(v || '').replace(/\\n/gi, ' ').replace(/\\([,;\\])/g, '$1').trim();
}

function _words(s) { return String(s || '').trim().split(/\s+/).filter(Boolean).length; }

// PHOTO is deliberately ignored — contacts render with the same initials
// avatar as every other connection.
function parseVcards(text) {
  const out = [];
  const cards = unfoldVcf(text).split(/BEGIN:VCARD/i).slice(1);
  for (const raw of cards) {
    const body = raw.split(/END:VCARD/i)[0];
    const c = { name:'', org:'', title:'', email:'', tel:'', sel:false, lastName:'' };
    let nameFromN = '';
    for (const line of body.split('\n')) {
      const ci = line.indexOf(':');
      if (ci < 0) continue;
      const segs  = line.slice(0, ci).split(';');
      const value = line.slice(ci + 1);
      // properties can carry a group prefix, e.g. "item1.TEL"
      let key = segs[0];
      const dot = key.indexOf('.');
      if (dot >= 0) key = key.slice(dot + 1);
      key = key.trim().toUpperCase();
      switch (key) {
        case 'FN': if (!c.name) c.name = vcfUnescape(value); break;
        case 'N': {
          // N: Family;Given;Middle;Prefix;Suffix
          const f = value.split(';').map(vcfUnescape);
          nameFromN = [f[3], f[1], f[2], f[0], f[4]].filter(Boolean).join(' ').trim();
          break;
        }
        case 'ORG':   if (!c.org)   c.org   = value.split(';').map(vcfUnescape).filter(Boolean).join(' · '); break;
        case 'TITLE': if (!c.title) c.title = vcfUnescape(value); break;
        case 'EMAIL': if (!c.email) c.email = vcfUnescape(value); break;
        case 'TEL':   if (!c.tel)   c.tel   = vcfUnescape(value); break;
      }
    }
    // Prefer whichever of FN / N carries a surname: iOS writes FN "Mom" next to
    // an N of "Kowalski;Mom", and the structured field is the fuller name.
    if (!c.name) c.name = nameFromN;
    else if (_words(c.name) < 2 && _words(nameFromN) > 1) c.name = nameFromN;
    if (c.name) out.push(c);
  }
  return out;
}

function vcfAffil(c) { return [c.title, c.org].filter(Boolean).join(' · '); }

// Phone books are full of one-name entries — "Mom", "Dave", "Plumber". A bare
// first name can't be matched against the graph and silently merges with every
// other "Dave", so those never ride along with the main list: they get their own
// step where you supply the surname or leave them out.
function vcfNeedsLastName(c) { return _words(c.name) < 2; }
function vcfFullName(c) {
  return c.needsLast ? `${c.name} ${c.lastName}`.trim() : c.name;
}
// what would actually be imported right now
function vcfPicked() {
  return vcfContacts.filter(c => c.needsLast ? c.lastName.trim() : c.sel);
}

function vcfDrop(e) {
  e.preventDefault();
  document.getElementById('vcfDrop').classList.remove('dragover');
  readVcfFiles(Array.from(e.dataTransfer.files || []));
}
function vcfFileChange(e) {
  readVcfFiles(Array.from(e.target.files || []));
  e.target.value = '';
}

function readVcfFiles(files) {
  if (!files.length) return;
  const hdr = document.getElementById('vcfPreviewHdr');
  document.getElementById('vcfPreview').style.display = 'block';
  hdr.textContent = 'reading…';
  Promise.all(files.map(f => f.text()))
    .then(texts => {
      let cards = [];
      texts.forEach(t => { cards = cards.concat(parseVcards(t)); });
      showVcfContacts(cards, files.length > 1 ? `${files.length} files` : files[0].name);
    })
    .catch(() => { hdr.textContent = "couldn't read that file"; });
}

async function pickDeviceContacts() {
  try {
    const picked = await navigator.contacts.select(['name', 'email', 'tel'], { multiple: true });
    showVcfContacts(picked.map(p => ({
      name:  (p.name  || []).filter(Boolean)[0] || '',
      org:'', title:'', sel:false,
      email: (p.email || []).filter(Boolean)[0] || '',
      tel:   (p.tel   || []).filter(Boolean)[0] || ''
    })).filter(c => c.name), 'device');
  } catch (e) {
    document.getElementById('vcfPreviewHdr').textContent = 'contact picker cancelled or blocked';
  }
}

function showVcfContacts(cards, label) {
  // de-dupe cards repeated across files / lists
  const seen = new Set();
  vcfContacts = cards.filter(c => {
    const k = c.name.toLowerCase() + '|' + (c.email || c.tel || '');
    if (seen.has(k)) return false;
    seen.add(k); return true;
  });
  vcfContacts.forEach(c => { c.needsLast = vcfNeedsLastName(c); c.lastName = ''; });

  document.getElementById('vcfPreview').style.display = 'block';
  if (!vcfContacts.length) {
    document.getElementById('vcfPreviewHdr').textContent =
      `no contacts found in ${label} — expected a .vcf / vCard file`;
    document.getElementById('vcfList').innerHTML = '';
    document.getElementById('vcfSelectBar').style.display = 'none';
    document.getElementById('vcfFixWrap').style.display = 'none';
    refreshNetImportBtn();
    return;
  }
  // a handful of cards is almost always "import all"; a whole address book is not
  const full = vcfContacts.filter(c => !c.needsLast);
  const preselect = full.length <= 25;
  const existing = new Set((db.contacts || []).map(c => c.name.toLowerCase()));
  full.forEach(c => { c.sel = preselect && !existing.has(c.name.toLowerCase()); });
  document.getElementById('vcfSelectBar').style.display = full.length ? '' : 'none';
  renderVcfList();
  renderVcfFixList();
}

// The surname step. Rows are inputs, so this renders once per file load and is
// never re-rendered on keystrokes — that would blow away the focused field.
function renderVcfFixList() {
  const wrap = document.getElementById('vcfFixWrap');
  const rows = vcfContacts.map((c, i) => ({ c, i })).filter(({ c }) => c.needsLast);
  wrap.style.display = rows.length ? '' : 'none';
  if (!rows.length) return;
  document.getElementById('vcfFixHdr').textContent =
    `${rows.length} contact${rows.length !== 1 ? 's' : ''} with no last name · 0 filled in`;
  document.getElementById('vcfFixList').innerHTML = rows.map(({ c, i }) => `
    <div class="vcf-fix-row">
      <span class="vfr-first">${esc(c.name)}</span>
      <input type="text" class="vfr-input" placeholder="last name — blank = skip"
             value="${esc(c.lastName)}" oninput="setVcfLastName(${i}, this.value)">
      ${vcfAffil(c) ? `<span class="vr-co">${esc(vcfAffil(c))}</span>` : ''}
    </div>`).join('');
}

function setVcfLastName(i, value) {
  if (!vcfContacts[i]) return;
  vcfContacts[i].lastName = value;
  const rows = vcfContacts.filter(c => c.needsLast);
  const filled = rows.filter(c => c.lastName.trim()).length;
  document.getElementById('vcfFixHdr').textContent =
    `${rows.length} contact${rows.length !== 1 ? 's' : ''} with no last name · ${filled} filled in`;
  updateVcfHdr();
}

function setAllVcf(on) {
  const q = (document.getElementById('vcfSearch').value || '').trim().toLowerCase();
  vcfContacts.forEach(c => {
    if (c.needsLast) return;   // those are governed by the surname field, not this
    if (!q || c.name.toLowerCase().includes(q) || vcfAffil(c).toLowerCase().includes(q)) c.sel = on;
  });
  renderVcfList();
}

// ticking a box only changes the count — re-rendering the list here would throw
// away the user's scroll position halfway down a long address book
function toggleVcf(i, on) {
  if (vcfContacts[i]) vcfContacts[i].sel = on;
  updateVcfHdr();
}

function updateVcfHdr() {
  const q = (document.getElementById('vcfSearch').value || '').trim();
  const full = vcfContacts.filter(c => !c.needsLast);
  const sel  = vcfPicked().length;
  document.getElementById('vcfPreviewHdr').textContent =
    `${full.length} contact${full.length !== 1 ? 's' : ''} with a full name · ${sel} to import` +
    (q ? ` · ${_vcfShownCount} matching filter` : '');
  refreshNetImportBtn();
}

function renderVcfList() {
  const q = (document.getElementById('vcfSearch').value || '').trim().toLowerCase();
  const existing = new Set((db.contacts || []).map(c => c.name.toLowerCase()));
  const shown = vcfContacts
    .map((c, i) => ({ c, i }))
    .filter(({ c }) => !c.needsLast)
    .filter(({ c }) => !q || c.name.toLowerCase().includes(q) || vcfAffil(c).toLowerCase().includes(q));

  _vcfShownCount = shown.length;
  updateVcfHdr();

  const MAX = 300;
  document.getElementById('vcfList').innerHTML = shown.slice(0, MAX).map(({ c, i }) => `
    <label class="vcf-row">
      <input type="checkbox" ${c.sel ? 'checked' : ''} onchange="toggleVcf(${i}, this.checked)">
      <span class="vr-name">${esc(c.name)}</span>
      ${vcfAffil(c) ? `<span class="vr-co">${esc(vcfAffil(c))}</span>` : ''}
      <span class="vr-dupe">${existing.has(c.name.toLowerCase()) ? 'ALREADY IN ARTEMIS' : ''}</span>
    </label>`).join('') +
    (shown.length > MAX
      ? `<div class="vcf-row" style="color:var(--ink-faint)">…and ${shown.length - MAX} more — use the filter</div>`
      : '');
}

async function confirmVcfImport() {
  const picked = vcfPicked();
  if (!picked.length) return;
  const btn = document.getElementById('liImportBtn');
  btn.disabled = true; const orig = btn.textContent; btn.textContent = 'IMPORTING…';
  try {
    const res = await fetch('/network/contacts/import', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contacts: picked.map(c => ({
          name: vcfFullName(c), company: c.org, title: c.title, email: c.email,
          notes: c.tel ? `Phone: ${c.tel}` : ''
        }))
      })
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    closeLinkedInImport();
    await loadContactsFromBackend();
    renderContacts();
    alert(`Imported: ${data.ingested.created} new, ${data.ingested.updated} updated, ${data.ingested.skipped} skipped.`);
  } catch (e) {
    alert('Import failed: ' + e.message);
  } finally { btn.textContent = orig; refreshNetImportBtn(); }
}

// ══════════════════════════════════════════════════════
// CONNECTIONS PICKER PANEL (board view — adds a My Connections
// contact onto the current board page as a plain node)
// ══════════════════════════════════════════════════════
function toggleCPPanel() {
  const panel = document.getElementById('cpPanel');
  if (panel.classList.contains('open')) { closeCPPanel(); }
  else { panel.classList.add('open'); renderCPList(); setTimeout(()=>document.getElementById('cpSearch').focus(),60); }
}
function closeCPPanel() {
  document.getElementById('cpPanel').classList.remove('open');
}
function renderCPList() {
  const q = (document.getElementById('cpSearch')?.value||'').toLowerCase().trim();
  const contacts = (db.contacts||[]).filter(c => !q || c.name.toLowerCase().includes(q) || (c.company||'').toLowerCase().includes(q) || (c.role||'').toLowerCase().includes(q));
  const listEl = document.getElementById('cpList');
  if (!contacts.length) {
    listEl.innerHTML = '<div class="cp-empty">' + (db.contacts?.length ? 'No matches.' : 'No contacts yet.<br>Add them in My Connections.') + '</div>';
    return;
  }
  const pg = currentPage();
  const onBoard = new Set((pg?.people||[]).map(p => p.name.toLowerCase()));
  listEl.innerHTML = contacts.map(c => {
    const ini2 = c.name.split(' ').map(w=>w[0]).join('').slice(0,2).toUpperCase();
    const avatar = c.photo
      ? `<div class="cp-avatar"><img src="${esc(c.photo)}" onerror="this.parentNode.textContent='${ini2}'"></div>`
      : `<div class="cp-avatar">${ini2}</div>`;
    const alreadyAdded = onBoard.has(c.name.toLowerCase());
    const btn = alreadyAdded
      ? `<button class="cp-add-btn added" disabled>✓ Added</button>`
      : `<button class="cp-add-btn" onclick="addContactToBoard('${c.id}')">+ Add</button>`;
    const sub = [c.role, c.company].filter(Boolean).join(' · ');
    return `<div class="cp-item">
      ${avatar}
      <div class="cp-info">
        <div class="cp-name">${esc(c.name)}</div>
        ${sub ? `<div class="cp-role">${esc(sub)}</div>` : ''}
      </div>
      ${btn}
    </div>`;
  }).join('');
}
function addContactToBoard(contactId) {
  const contact = (db.contacts||[]).find(c => c.id === contactId);
  if (!contact) return;
  const pg = currentPage(); if (!pg) return;
  const w = document.getElementById('wrapper');
  const cx = (w.clientWidth/2 - panX) / zoom;
  const cy = (w.clientHeight/2 - panY) / zoom;
  const j = () => (Math.random() - 0.5) * 200;
  pg.people.push({
    id: uid(), name: contact.name, role: contact.role || '', photo: contact.photo || '',
    description: contact.description || '', size: 1,
    x: Math.round(cx + j()), y: Math.round(cy + j()),
  });
  save(); render();
  renderCPList();
}

// ══════════════════════════════════════════════════════
// GLOBAL KEY HANDLERS
// ══════════════════════════════════════════════════════
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    if (document.getElementById('ctLinkScrim')?.classList.contains('open'))   { closeLinkContactModal(); return; }
    if (document.getElementById('ctAddScrim')?.classList.contains('open'))    { closeAddContactModal(); return; }
    if (document.getElementById('ctRail')?.classList.contains('open'))        { closeContactRail(); return; }
    if (document.getElementById('bvRoutePanel')?.classList.contains('open'))  { closeBoardRouteFinder(); return; }
    if (document.getElementById('discoverScrim')?.classList.contains('open')){ closeDiscoverModal(); return; }
    if (document.getElementById('hvImportScrim')?.classList.contains('open')) { closeImportPreview(); return; }
    if (document.getElementById('hvModalScrim')?.classList.contains('open'))  { closeCreateModal(); return; }
    if (document.getElementById('hvRail')?.classList.contains('open'))        { closeDetailRail(); return; }
  }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'b' && !isTyping()) {
    e.preventDefault();
    if (document.getElementById('homeView').style.display !== 'flex') goHome(); else showHome();
  }
});

// ══════════════════════════════════════════════════════
// PROVIDER STATUS — search providers (Serper/Brave) are configured server-side
// only (platform secrets, e.g. `fly secrets set`); there is nothing for a user
// to enter here. This just hides the "not configured" notice the instant
// either key is present, so it's never shown once the operator has set one up.
// ══════════════════════════════════════════════════════
async function checkProviderStatus() {
  const warn = document.getElementById('hvProviderWarn');
  if (!warn) return;
  try {
    const s = await (await fetch('/status')).json();
    const configured = s?.serper?.state !== 'not_configured' || s?.brave?.state !== 'not_configured';
    warn.style.display = configured ? 'none' : '';
  } catch {
    warn.style.display = 'none';
  }
}

// ══════════════════════════════════════════════════════
// NETWORK GATE
// Artemis routes THROUGH the operator's own contacts. With none loaded it can
// still map a target's public network, but every route it returns terminates
// at a stranger — so an empty network is a correctness problem, not a missing
// nice-to-have, and the operator should be told before their first search
// rather than after it disappoints them.
//
// The boxes here deliberately do NOT reimplement parsing. They hand the file
// straight to the existing import pipeline (processLinkedInFile / readVcfFiles)
// and open the import modal on the matching tab, so the operator lands in the
// same preview → de-dupe → confirm flow as the in-app importer, and there is
// exactly one CSV parser and one vCard parser in the codebase.
// ══════════════════════════════════════════════════════
const BOOT_SPLASH_MS = 2100;   // #hvBoot: 1.6s hold + 0.5s fade (see index.html)

function maybeShowNetworkGate() {
  if (db.contacts.length) return;          // already has a network — nothing to warn about
  document.getElementById('hvGate')?.classList.add('open');
}

function dismissNetworkGate() {
  document.getElementById('hvGate')?.classList.remove('open');
}

// Hide-only, and called from loadContactsFromBackend so every path that can
// populate contacts (either importer, a manual add) closes the gate without
// each one having to remember to.
function syncNetworkGate() {
  if (db.contacts.length) dismissNetworkGate();
}

function _gateHandoff(tab) {
  // Close the gate and open the real importer: the operator still gets the
  // preview and confirm step, and a cancel there leaves them in the app rather
  // than trapped back behind the gate.
  dismissNetworkGate();
  showLinkedInImport();
  switchNetImportTab(tab);
}

function gateDropCsv(e) {
  e.preventDefault();
  document.getElementById('hgBoxCsv').classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (!file) return;
  _gateHandoff('csv');
  processLinkedInFile(file);
}

function gateCsvChange(e) {
  const file = e.target.files[0];
  e.target.value = '';                     // re-picking the same file must re-fire
  if (!file) return;
  _gateHandoff('csv');
  processLinkedInFile(file);
}

function gateDropVcf(e) {
  e.preventDefault();
  document.getElementById('hgBoxVcf').classList.remove('dragover');
  const files = Array.from(e.dataTransfer.files || []);
  if (!files.length) return;
  _gateHandoff('vcf');
  readVcfFiles(files);
}

function gateVcfChange(e) {
  const files = Array.from(e.target.files || []);
  e.target.value = '';
  if (!files.length) return;
  _gateHandoff('vcf');
  readVcfFiles(files);
}

// ══════════════════════════════════════════════════════
// BOOT
// ══════════════════════════════════════════════════════
(async function boot() {
  const started = Date.now();
  setOperatorName(operatorName());
  await Promise.all([loadBoardsFromBackend(), loadContactsFromBackend()]);
  showHome();
  checkProviderStatus();
  // Wait out whatever is left of the splash. Loading usually finishes first,
  // and showing the gate early would stack it on top of the boot animation.
  setTimeout(maybeShowNetworkGate, Math.max(0, BOOT_SPLASH_MS - (Date.now() - started)));
})();
