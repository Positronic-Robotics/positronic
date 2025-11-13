document.addEventListener('DOMContentLoaded', () => {
  const sidebarState = loadSidebarState();
  const sidebarToggler = document.querySelector('.sidebar-toggler');
  const sidebar = document.querySelector('.sidebar');
  const sidebarResizer = document.querySelector('.sidebar-resizer');
  const sidebarContent = document.querySelector('.sidebar-content');
  const keyColumnResizer = document.querySelector('.key-column-resizer');
  const episodeViewer = document.getElementById('viewer-container');

  setSidebarWidth(sidebarState.sidebarWidth);
  setKeyColumnWidth(sidebarState.keyColumnWidth);
  setSidebarScrollTop(sidebarState.scrollTop);
  toggleSidebar(sidebarState.isExpanded);

  const sidebarResizeHandler = (event) => {
    event.preventDefault();
    setSidebarWidth(document.body.clientWidth - event.clientX);
    saveSidebarState();
  };

  const keyColumnResizeHandler = (event) => {
    const offsetX = sidebarState.sidebarWidth - (document.body.clientWidth - event.clientX);
    const columnWidth = Math.min(Math.max(50, offsetX), sidebarState.sidebarWidth - 50);
    setKeyColumnWidth(columnWidth);
    saveSidebarState();
  };

  sidebarToggler.addEventListener('click', () => {
    const isExpanded = sidebar.classList.contains('expanded');
    toggleSidebar(!isExpanded);
    saveSidebarState();
  });

  [
    { resizer: sidebarResizer, handler: sidebarResizeHandler },
    { resizer: keyColumnResizer, handler: keyColumnResizeHandler },
  ].forEach(({ resizer, handler }) => {
    resizer.addEventListener('mousedown', (event) => {
      episodeViewer.style.zIndex = -1;
      sidebar.classList.add('isResizing');
      document.body.style.userSelect = 'none';
      document.addEventListener('mousemove', handler);

      document.addEventListener(
        'mouseup',
        (event) => {
          document.removeEventListener('mousemove', handler);
          episodeViewer.style.zIndex = '';
          sidebar.classList.remove('isResizing');
          document.body.style.userSelect = '';
        },
        { once: true }
      );
    });
  });

  sidebarContent.addEventListener('scroll', () => {
    const scrollTop = sidebarContent.scrollTop;
    setSidebarScrollTop(scrollTop);
    saveSidebarState();
  });

  function setSidebarWidth(width) {
    sidebarState.sidebarWidth = Math.max(100, width);
    sidebar.style.width = `${sidebarState.sidebarWidth}px`;
  }

  function toggleSidebar(isExpanded) {
    sidebar.classList.toggle('expanded', isExpanded);
    sidebar.style.width = isExpanded ? `${sidebarState.sidebarWidth}px` : '0px';
    sidebarState.isExpanded = isExpanded;
  }

  function setKeyColumnWidth(width) {
    sidebar.style.setProperty('--dataset-key-column-width', `${width}px`);
    sidebarState.keyColumnWidth = width;
  }

  function setSidebarScrollTop(scrollTop) {
    sidebarContent.scrollTop = scrollTop;
    sidebarState.scrollTop = scrollTop;
  }

  function saveSidebarState() {
    localStorage.setItem('sidebarState', JSON.stringify(sidebarState));
  }

  function loadSidebarState() {
    const defaultState = {
      isExpanded: false,
      sidebarWidth: 300,
      keyColumnWidth: 150,
      scrollTop: 0,
    };
    const savedState = JSON.parse(localStorage.getItem('sidebarState'));

    return { ...defaultState, ...savedState };
  }
});

function initializeSidebar(staticData) {
  const sidebarContent = document.querySelector('.sidebar-content-wrapper');

  function isNestable(value) {
    return typeof value === 'object' && value !== null;
  }

  function renderValue(value, level = 0) {
    const expandButton = level === 0 ? '<button class="expand-button">▶</button>' : '';

    switch (typeof value) {
      case 'string':
        return `"${value}"`;
      case 'number':
        return value.toFixed(2).replace(/\.00$/, '');
      case 'boolean':
        return `<input type="checkbox" ${value ? 'checked' : ''} onclick="return false" />`;

      case 'object':
        if (value === null) {
          return 'null';
        } else if (Array.isArray(value)) {
          return expandButton + `[${value.map((item) => renderValue(item, level + 1)).join(', ')}]`;
        } else {
          return expandButton + '{...}';
        }
      default:
        return value;
    }
  }

  function renderLevel(key, value, isRoot = false) {
    let html = isRoot
      ? '<table class="root-table"><colgroup><col class="dataset-key-column"><col></colgroup>'
      : '<table>';
    html += '<tbody>';

    if (isNestable(value)) {
      for (const [key, val] of Object.entries(value)) {
        html += `
          <tr>
            <td>${key}</td>
              <td>${renderValue(val)} ${isNestable(val) ? renderLevel(key, val) : ''}</td>
          </tr>
        `;
      }
    } else {
      html += `<tr><td>${key}</td><td>${renderValue(value)}</td></tr>`;
    }
    html += '</tbody></table>';

    return html;
  }

  sidebarContent.insertAdjacentHTML('beforeend', renderLevel('', staticData, true));
  document.querySelectorAll('.expand-button').forEach((button) => {
    button.addEventListener('click', () => {
      const table = button.parentElement.querySelector('table');

      if (table) {
        const isShown = button.dataset.expanded === 'true';
        button.dataset.expanded = !isShown;
        table.style.display = isShown ? 'none' : 'table';
      }
    });
  });
}
