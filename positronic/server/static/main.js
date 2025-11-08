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
