const filtersState = {
  sort: { columnIndex: null, direction: 'desc' },
  filters: {},
  serverFilters: {},
};
let filtersData = {};
let loadingCheckInterval = null;
let currentEpisodes = [];

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

      const { episodes, columns, group_filters: groupFilters } = await loadEpisodes();
      currentEpisodes = episodes;
      filtersData = getFiltersData(episodes, columns);
      renderServerFilters(groupFilters);
      renderClientFilters(columns);
      renderEpisodesTableHeader(columns);
      populateEpisodesTable(columns);
      episodesLoadingStatus.remove();
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

async function loadEpisodes(filters = {}) {
  try {
    const endpoint = new URL(window.API_ENDPOINT || '/api/episodes', window.location.origin);

    Object.entries(filters).forEach(([key, value]) => {
      endpoint.searchParams.append(key, value);
    });

    const response = await fetch(endpoint);
    if (response.status === 202) {
      checkDatasetStatus();
      return;
    }

    return await response.json();
  } catch (error) {
    console.error('Error loading episodes:', error);
    document.getElementById('episodes-container').innerHTML =
      '<div class="loading">Error loading episodes</div>';
  }
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

function renderEpisodesTableHeader(columns) {
  const headerRow = document.querySelector('.episodes-table thead tr');
  const headerColumns = [];
  const sortState = filtersState.sort;

  for (const [columnIndex, { label }] of Object.entries(columns)) {
    const headerColumn = createTableCell(label, true);
    headerColumn.classList.add('sortable');
    headerColumn.dataset.sort = columnIndex;
    headerColumns.push(headerColumn);

    headerColumn.addEventListener('click', sortByColumnHandler);
  }

  headerRow.prepend(...headerColumns);

  function sortByColumnHandler(event) {
    const headerColumn = event.currentTarget;
    const columnIndex = headerColumn.dataset.sort;
    headerRow.querySelectorAll('th.sortable').forEach((th) => {
      th.classList.remove('sorted-asc', 'sorted-desc');
    });

    if (sortState.columnIndex === columnIndex) {
      sortState.direction = sortState.direction === 'desc' ? 'asc' : 'desc';
    } else {
      sortState.columnIndex = columnIndex;
      sortState.direction = 'desc';
    }

    headerColumn.classList.add(`sorted-${sortState.direction}`);

    populateEpisodesTable(columns);
  }
}

function renderServerFilters(groupFilters) {
  if (!groupFilters) return;
  const controlsBar = document.querySelector('.controls-bar');

  for (const [filterKey, filterData] of Object.entries(groupFilters)) {
    const options = [createFilterOption('-1', 'All')];

    for (const value of filterData.values) {
      options.push(createFilterOption(value, value));
    }

    const filterContainer = createFilter({
      filterId: `filter-${filterKey}`,
      label: filterData.label,
      options: options,
      onChange: async (event) => {
        if (event.target.value === '-1') {
          delete filtersState.serverFilters[filterKey];
        } else {
          filtersState.serverFilters[filterKey] = event.target.value;
        }

        const { episodes, columns } = await loadEpisodes(filtersState.serverFilters);
        currentEpisodes = episodes;
        populateEpisodesTable(columns);
      },
    });

    controlsBar.appendChild(filterContainer);
  }
}

function renderClientFilters(columns) {
  const controlsBar = document.querySelector('.controls-bar');

  for (const [filterIndex, filter] of Object.entries(filtersData)) {
    const options = [createFilterOption('-1', 'All')];

    for (const [index, value] of filter.entries()) {
      const label = columns[filterIndex].renderer?.options[value]?.label ?? value;
      options.push(createFilterOption(index, label));
    }

    const filterContainer = createFilter({
      filterId: `filter-${filterIndex}`,
      label: columns[filterIndex].label,
      options: options,
      onChange: (event) => {
        if (event.target.value === '-1') {
          delete filtersState.filters[filterIndex];
        } else {
          filtersState.filters[filterIndex] = event.target.value;
        }

        populateEpisodesTable(columns);
      },
    });

    controlsBar.appendChild(filterContainer);
  }
}

function createFilter({ filterId, label, options, onChange }) {
  const filterContainer = document.createElement('div');
  filterContainer.classList.add('control-group', 'control-group--grow');

  const labelElement = document.createElement('label');
  labelElement.htmlFor = filterId;
  labelElement.textContent = label;
  filterContainer.appendChild(labelElement);

  const select = document.createElement('select');
  select.id = filterId;
  select.addEventListener('change', onChange);
  select.append(...options);

  filterContainer.appendChild(select);

  return filterContainer;
}

function createFilterOption(value, label) {
  const option = document.createElement('option');
  option.value = value;
  option.textContent = label;
  return option;
}

function getFiltersData(episodes, columns) {
  const filtersData = {};

  for (const [index, column] of Object.entries(columns)) {
    if (!column.filter) continue;

    const columnData = new Set();

    for (const [episodeIndex, episodeData] of episodes) {
      const value = episodeData[index];
      if (value === null || value === undefined) continue;

      columnData.add(String(value));
    }

    filtersData[index] = Array.from(columnData);
  }

  return filtersData;
}

function populateEpisodesTable(columns) {
  const tableBody = document.querySelector('.episodes-table tbody');
  const filteredEpisodes = getFilteredEpisodes(currentEpisodes);

  tableBody.innerHTML = '';
  for (const [episodeIndex, episodeData] of filteredEpisodes) {
    const row = document.createElement('tr');

    for (const [index, entity] of episodeData.entries()) {
      row.appendChild(createTableCell(getCellValue(entity, columns[index])));
    }

    const viewCell = createTableCell('');
    if (window.SHOW_VIEW_LINK !== false) {
      const viewLink = document.createElement('a');
      viewLink.className = 'btn btn-primary btn-small';
      viewLink.href = `/episode/${episodeIndex}`;
      viewLink.textContent = 'View';
      viewCell.appendChild(viewLink);
    }
    row.appendChild(viewCell);

    tableBody.appendChild(row);
  }

  function getCellValue(entity, column) {
    if (column.renderer?.type === 'badge') {
      return createBadge(entity, column.renderer.options?.[entity]);
    }

    const value = Array.isArray(entity) ? entity[1] : entity;

    return value ?? '';
  }

  function createBadge(value, options) {
    if (value === null || value === undefined) return '';

    const allowedVariants = ['success', 'warning', 'danger', 'default'];
    const span = document.createElement('span');
    span.classList.add('badge');
    span.textContent = options?.label ?? String(value);
    let variantClass;

    if (options?.variant && allowedVariants.includes(options.variant)) {
      variantClass = options.variant;
    } else {
      variantClass = 'default';
    }

    span.classList.add(`badge--${variantClass}`);

    return span;
  }
}

function getFilteredEpisodes(episodes) {
  const { sort: sortState, filters } = filtersState;
  let filteredEpisodes = episodes.slice();

  filteredEpisodes = filteredEpisodes.filter(([episodeIndex, episodeData]) => {
    return Object.entries(filters).every(([columnIndex, valueIndex]) => {
      return String(episodeData[columnIndex]) === filtersData[columnIndex][valueIndex];
    });
  });

  if (sortState.columnIndex === null) {
    return filteredEpisodes;
  }

  filteredEpisodes.sort((a, b) => {
    const aValue = getSortableValue(a[1][sortState.columnIndex]);
    const bValue = getSortableValue(b[1][sortState.columnIndex]);

    if (aValue === bValue) return 0;

    if (sortState.direction === 'asc') {
      return aValue < bValue ? -1 : 1;
    } else {
      return aValue > bValue ? -1 : 1;
    }
  });

  return filteredEpisodes;

  function getSortableValue(entity) {
    if (entity === null || entity === undefined) return '';

    const value = Array.isArray(entity) ? entity[0] : entity;

    return value;
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
