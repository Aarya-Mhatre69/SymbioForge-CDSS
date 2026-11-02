(function () {
  'use strict';
  document.documentElement.classList.add('js');
  var KEY = 'meridian_bookings';

  // Embedded app: set iframe src and fallback link from data-app-url.
  var frame = document.getElementById('app');
  if (frame) {
    var url = frame.getAttribute('data-app-url');
    frame.src = url;
    var link = document.getElementById('app-link');
    if (link) link.href = url;
  }

  // Clinician link: base url + &view=clinician
  var cl = document.getElementById('clinician-link');
  if (cl) {
    var base = cl.getAttribute('data-app-url');
    var q = cl.getAttribute('data-app-query');
    cl.href = base + (base.indexOf('?') === -1 ? '?' : '&') + q;
  }

  // Bookings
  var form = document.getElementById('booking-form');
  if (!form) return;
  var list = document.getElementById('booking-list');
  var conf = document.getElementById('confirmation');

  function load() {
    try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch (e) { return []; }
  }
  function save(items) {
    try { localStorage.setItem(KEY, JSON.stringify(items)); return true; } catch (e) { return false; }
  }
  function render() {
    var items = load();
    list.textContent = '';
    if (!items.length) {
      var li = document.createElement('li');
      li.className = 'muted';
      li.textContent = 'No demo bookings stored yet.';
      list.appendChild(li);
      return;
    }
    items.forEach(function (b) {
      var li = document.createElement('li');
      var s = document.createElement('strong');
      s.textContent = b.name;
      li.appendChild(s);
      li.appendChild(document.createElement('br'));
      var small = document.createElement('small');
      small.textContent = b.date + ', ' + b.window + ' · ' + b.phone + (b.address ? ' · ' + b.address : '');
      li.appendChild(small);
      list.appendChild(li);
    });
  }
  function setErr(id, msg) {
    document.getElementById(id + '-err').textContent = msg || '';
    var el = document.getElementById(id);
    if (msg) el.setAttribute('aria-invalid', 'true'); else el.removeAttribute('aria-invalid');
    return !msg;
  }

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    var f = form.elements;
    var name = f.name.value.trim(), phone = f.phone.value.trim();
    var ok = [
      setErr('name', name ? '' : 'Please enter a name.'),
      setErr('phone', /^[0-9+\-()\s]{7,}$/.test(phone) ? '' : 'Please enter a phone number, e.g. 000-000-0000.'),
      setErr('date', f.date.value ? '' : 'Please choose a date.'),
      setErr('window', f.window.value ? '' : 'Please choose a time window.'),
      setErr('consent', f.consent.checked ? '' : 'Please tick the box to continue.')
    ].every(Boolean);
    if (!ok) {
      var first = form.querySelector('[aria-invalid="true"]');
      if (first) first.focus();
      return;
    }
    var items = load();
    items.push({ name: name, phone: phone, date: f.date.value, window: f.window.value, address: f.address.value.trim(), saved: new Date().toISOString() });
    var saved = save(items);
    conf.hidden = false;
    conf.className = 'note ok';
    conf.textContent = saved
      ? 'Demo only: nothing was sent. Your request is listed below, stored in this browser.'
      : 'Demo only: nothing was sent. This browser blocked local storage, so the request was not listed.';
    form.reset();
    conf.focus();
    render();
  });

  document.getElementById('clear-bookings').addEventListener('click', function () {
    try { localStorage.removeItem(KEY); } catch (e) {}
    conf.hidden = true;
    render();
  });

  render();
})();
