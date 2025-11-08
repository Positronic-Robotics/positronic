document.addEventListener('DOMContentLoaded', () => {
  let sidebarWidth = 300;
  const sidebarToggler = document.querySelector('.sidebar-toggler');
  const sidebar = document.querySelector('.sidebar');
  const sidebarResizer = document.querySelector('.sidebar-resizer');
  const episodeViewer = document.getElementById('viewer-container');

  const sidebarResizeHandler = (event) => {
    event.preventDefault();
    sidebarWidth = Math.max(100, document.body.clientWidth - event.clientX);
    sidebar.style.width = `${sidebarWidth}px`;
  };

  sidebarToggler.addEventListener('click', () => {
    const isExpanded = sidebar.classList.toggle('expanded');
    sidebar.style.width = isExpanded ? `${sidebarWidth}px` : '0px';
  });

  sidebarResizer.addEventListener('mousedown', (event) => {
    episodeViewer.style.zIndex = -1;
    sidebar.classList.add('isResizing');
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', sidebarResizeHandler);

    document.addEventListener(
      'mouseup',
      (event) => {
        document.removeEventListener('mousemove', sidebarResizeHandler);
        episodeViewer.style.zIndex = '';
        sidebar.classList.remove('isResizing');
        document.body.style.userSelect = '';
      },
      { once: true }
    );
  });
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

  function renderLevel(key, value) {
    let html = `<table><tbody>`;

    if (isNestable(value)) {
      for (const [key, val] of Object.entries(value)) {
        html += `
          <tr>
            <td><strong>${key}</strong></td>
              <td>${renderValue(val)} ${isNestable(val) ? renderLevel(key, val) : ''}</td>
          </tr>
        `;
      }
    } else {
      html += `<tr><td><strong>${key}</strong></td><td>${renderValue(value)}</td></tr>`;
    }
    html += '</tbody></table>';

    return html;
  }

  sidebarContent.innerHTML = renderLevel('', staticData);
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
