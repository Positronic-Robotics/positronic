let loadingCheckInterval = null;

async function checkDatasetStatus() {
  try {
    const response = await fetch('/api/dataset_status');
    const status = await response.json();

    const datasetStats = document.getElementById('dataset-stats');
    const episodesContainer = document.getElementById('episodes-container');
    const episodesTable = episodesContainer.querySelector('.episodes-table');
    const datasetLoadingStatus = document.getElementById('loading-status');
    const episodesLoadingStatus = episodesContainer.querySelector('.loading');

    if (status.loading) {
      datasetLoadingStatus.classList.add('show');
      datasetStats.innerHTML = 'Checking dataset status...';
      episodesContainer.innerHTML = '<div class="loading">Waiting for dataset to load...</div>';

      if (!loadingCheckInterval) {
        loadingCheckInterval = setInterval(checkDatasetStatus, 2000);
      }
    } else if (status.loaded) {
      if (loadingCheckInterval) {
        clearInterval(loadingCheckInterval);
        loadingCheckInterval = null;
      }
      datasetLoadingStatus.classList.remove('show');
      loadDatasetInfo();

      const { episodes, columns } = await loadEpisodes();
      populateEpisodesTable(episodes, columns);
      episodesContainer.removeChild(episodesLoadingStatus);
      episodesTable.classList.remove('hidden');
    } else {
      if (loadingCheckInterval) {
        clearInterval(loadingCheckInterval);
        loadingCheckInterval = null;
      }
      datasetLoadingStatus.classList.remove('show');
      datasetStats.innerHTML = '<span class="error-message">Failed to load dataset</span>';
      episodesContainer.innerHTML =
        '<div class="loading error-message">Dataset loading failed</div>';
    }
  } catch (error) {
    console.error('Error checking dataset status:', error);
  }
}

async function loadDatasetInfo() {
  try {
    const response = await fetch('/api/dataset_info');
    if (response.status === 202) {
      checkDatasetStatus();
      return;
    }

    const data = await response.json();

    const statsDiv = document.getElementById('dataset-stats');
    statsDiv.innerHTML = `<p><strong>${data.num_episodes}</strong> episodes.</p>`;
  } catch (error) {
    console.error('Error loading dataset info:', error);
  }
}

async function loadEpisodes() {
  try {
    const response = await fetch('/api/episodes');
    if (response.status === 202) {
      checkDatasetStatus();
      return;
    }

    const { episodes, columns } = await response.json();

    return { episodes, columns };
  } catch (error) {
    console.error('Error loading episodes:', error);
    document.getElementById('episodes-container').innerHTML =
      '<div class="loading">Error loading episodes</div>';
  }
}

function populateEpisodesTable(episodes, columns) {
  const headerRow = document.querySelector('.episodes-table thead tr');
  const tableBody = document.querySelector('.episodes-table tbody');
  const renderers = [];
  const headerColumns = [];

  for (const { label, renderer } of Object.values(columns)) {
    headerColumns.push(createTableCell(label, true));
    renderers.push(renderer ?? null);
  }

  headerRow.prepend(...headerColumns);

  for (const episode of episodes) {
    const row = document.createElement('tr');

    for (const [index, value] of episode.entries()) {
      row.appendChild(createTableCell(index === 0 ? value : getValue(value, renderers[index])));
    }

    const viewCell = createTableCell('');
    const viewLink = document.createElement('a');
    viewLink.className = 'btn btn-primary btn-small';
    viewLink.href = `/episode/${episode[0]}`; // Assuming the first element is the episode index or ID
    viewLink.textContent = 'View';
    viewCell.appendChild(viewLink);
    row.appendChild(viewCell);

    tableBody.appendChild(row);
  }

  function createTableCell(content, isHeader = false) {
    const episodeCell = document.createElement(isHeader ? 'th' : 'td');

    if (content instanceof HTMLElement) {
      episodeCell.appendChild(content);
    } else {
      episodeCell.textContent = String(content);
    }

    return episodeCell;
  }

  function getValue(value, renderer) {
    if (renderer?.type == 'badge') {
      return createBadge(value, renderer.options?.[value]);
    }

    switch (typeof value) {
      case 'number':
        return value.toFixed(2);
      default:
        return String(value ?? 'N/A');
    }
  }

  function createBadge(value, options) {
    if (value === null || value === undefined) return '';

    const span = document.createElement('span');
    span.classList.add('badge');
    span.textContent = options?.label ?? String(value);

    let variant;
    switch (options?.variant) {
      case 'success':
        variant = 'badge--success';
        break;

      case 'warning':
        variant = 'badge--warning';
        break;

      case 'danger':
        variant = 'badge--danger';
        break;

      default:
        variant = 'badge--default';
    }

    span.classList.add(variant);

    return span;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const sidebarState = loadSidebarState();
  const sidebarToggler = document.querySelector('.sidebar-toggler');
  const sidebar = document.querySelector('.sidebar');
  const sidebarResizer = document.querySelector('.sidebar-resizer');
  const sidebarContent = document.querySelector('.sidebar-content');
  const keyColumnResizer = document.querySelector('.key-column-resizer');
  const episodeViewer = document.getElementById('viewer-container');

  // TODO: Split logic to run page-specific explicitly
  if (!sidebar) return;

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
  const sidebarContent = document.querySelector('.sidebar-content-wrapper tbody');

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
    let html = '';

    if (isNestable(value)) {
      for (const [key, val] of Object.entries(value)) {
        html += `
          <tr>
            <td>${key}</td>
              <td>
                ${renderValue(val)}
                ${isNestable(val) ? `<table><tbody>${renderLevel(key, val)}</tbody></table>` : ''}
              </td>
          </tr>
        `;
      }
    } else {
      html += `<tr><td>${key}</td><td>${renderValue(value)}</td></tr>`;
    }

    return html;
  }

  sidebarContent.insertAdjacentHTML('beforeend', renderLevel('', staticData));
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
