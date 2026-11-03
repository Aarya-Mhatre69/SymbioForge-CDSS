// Single place to point the clinic site at the SymbioForge app.
// Hosted clinic site -> the deployed Streamlit app. Opened locally (localhost / file) -> the local app started by run_demo.py.
(function () {
  var local = /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname) || location.protocol === 'file:';
  window.SYMBIOFORGE_APP_URL = local ? 'http://localhost:8501' : 'https://symbioforge.streamlit.app';
})();
