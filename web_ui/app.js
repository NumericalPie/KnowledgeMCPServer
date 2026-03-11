const el = id => document.getElementById(id);

el('indexUrlBtn').addEventListener('click', async () => {
  const url = el('urlInput').value.trim();
  if (!url) return alert('Enter a URL');
  el('indexUrlStatus').textContent = 'Indexing...';
  try {
    const res = await fetch('/mcp/index_url', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({url})
    });
    const data = await res.json();
    el('indexUrlStatus').textContent = JSON.stringify(data);
  } catch (e) {
    el('indexUrlStatus').textContent = 'Error: '+e;
  }
});

el('indexPdfBtn').addEventListener('click', async () => {
  const fileInput = el('pdfInput');
  if (!fileInput.files.length) return alert('Select a PDF file');
  const file = fileInput.files[0];
  el('indexPdfStatus').textContent = 'Indexing...';
  try {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch('/mcp/index_pdf', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    el('indexPdfStatus').textContent = JSON.stringify(data);
    fileInput.value = ''; // Clear file input
  } catch (e) {
    el('indexPdfStatus').textContent = 'Error: '+e;
  }
});

el('indexTexBtn').addEventListener('click', async () => {
  const fileInput = el('texInput');
  if (!fileInput.files.length) return alert('Select a TeX file');
  const file = fileInput.files[0];
  el('indexTexStatus').textContent = 'Indexing...';
  try {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch('/mcp/index_tex', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    el('indexTexStatus').textContent = JSON.stringify(data);
    fileInput.value = ''; // Clear file input
  } catch (e) {
    el('indexTexStatus').textContent = 'Error: '+e;
  }
});

el('addMdBtn').addEventListener('click', async () => {
  const markdown = el('mdText').value;
  const title = el('mdTitle').value;
  const repo = el('mdRepo').value;
  if (!markdown.trim()) return alert('Enter markdown text');
  el('addMdStatus').textContent = 'Adding...';
  try {
    const res = await fetch('/mcp/documents', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({repo,title,markdown})
    });
    const data = await res.json();
    el('addMdStatus').textContent = JSON.stringify(data);
  } catch(e) { el('addMdStatus').textContent = 'Error: '+e }
});

el('queryBtn').addEventListener('click', async () => {
  const q = el('qInput').value;
  const top_k = parseInt(el('topK').value||3,10);
  el('queryResults').textContent = 'Querying...';
  try {
    const res = await fetch('/mcp/query', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({q, top_k})
    });
    const data = await res.json();
    const out = data.results.map(r => `<div class="card"><strong>${r.id}</strong> score: ${r.score.toFixed(3)}<pre>${(r.doc_text||'').slice(0,800)}</pre></div>`).join('\n');
    el('queryResults').innerHTML = out || 'No results';
  } catch(e) { el('queryResults').textContent = 'Error: '+e }
});

el('listDocsBtn').addEventListener('click', async ()=>{
  el('docsList').textContent = 'Loading...';
  try{
    const res = await fetch('/mcp/documents');
    const data = await res.json();
    const keys = Object.keys(data||{});
    if(!keys.length) return el('docsList').textContent = 'No documents stored';
    el('docsList').innerHTML = keys.map(k=>`<div class="card"><strong>${k}</strong><pre>${(data[k].path||'')}</pre></div>`).join('\n');
  }catch(e){ el('docsList').textContent='Error: '+e }
});
