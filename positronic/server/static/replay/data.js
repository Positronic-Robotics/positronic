// The list a <positronic-replay> shows: a manifest or the host's own rows, read into one shape.
//
// A list is {title, about, columns, rows}. Each row carries its cell values and the URLs of its media, all
// absolute, so the element loads a media file only where a row names one.

// A manifest names no columns for the index and the duration of an episode, so the list adds them.
// Their keys start with `$`.
export const INDEX_KEY = '$index';
export const DURATION_KEY = '$duration';

export const ColumnType = Object.freeze({ TEXT: 'text', NUMBER: 'number', BADGE: 'badge', ICON: 'icon' });
const COLUMN_TYPES = new Set(Object.values(ColumnType));

/** A cell value as text: an object's `text`, else the value; null and undefined are empty. */
export function cellText(value) {
  if (value !== null && typeof value === 'object') return cellText(value.text);
  return value === null || value === undefined ? '' : String(value);
}

/** The value a filter compares: an object's `key`, else its text. */
export function cellKey(value) {
  if (value !== null && typeof value === 'object' && value.key !== undefined) return String(value.key);
  return cellText(value);
}

/** A media URL: a string, or an object with `src` (relative to `base`) or `url`. Null where the value is empty. */
function mediaUrl(value, base) {
  if (value === null || value === undefined || value === '') return null;
  const spelled = typeof value === 'object' ? (value.src ?? value.url) : value;
  return spelled ? new URL(spelled, base).href : null;
}

function readCamera(camera, base) {
  return {
    camera: String(camera.camera),
    label: camera.label === undefined ? null : String(camera.label),
    src: mediaUrl(camera, base),
    width: Number(camera.width) || null,
    height: Number(camera.height) || null,
    durationS: Number(camera.duration_s) || 0,
  };
}

function readFilter(filter) {
  if (!filter) return null;
  const spelled = filter === true ? {} : filter;
  const options = spelled.options?.map((option) =>
    Array.isArray(option)
      ? { value: String(option[0]), label: String(option[1]) }
      : { value: String(option.value), label: String(option.label ?? option.value) },
  );
  return { all: spelled.all === undefined ? 'All' : String(spelled.all), options: options ?? null };
}

function readColumn(column) {
  const type = column.type ?? ColumnType.TEXT;
  if (!COLUMN_TYPES.has(type)) throw new Error(`column ${column.key} has an unknown type ${type}`);
  return {
    key: String(column.key),
    label: String(column.label ?? column.key),
    type,
    decimals: column.decimals === undefined ? null : Number(column.decimals),
    unit: column.unit === undefined ? '' : String(column.unit),
    hover: column.hover === undefined ? null : String(column.hover),
    filter: readFilter(column.filter),
  };
}

/** Every value of a filtered column, in the order of the rows, where the host names no options. */
function withOptions(columns, rows) {
  return columns.map((column) => {
    if (!column.filter || column.filter.options) return column;
    const seen = new Map();
    for (const row of rows) {
      const value = row.values[column.key];
      const key = cellKey(value);
      if (key && !seen.has(key)) seen.set(key, cellText(value));
    }
    const options = [...seen].map(([value, label]) => ({ value, label }));
    return { ...column, filter: { ...column.filter, options } };
  });
}

function readRow(row, base, fallbackId) {
  return {
    id: String(row.id ?? fallbackId),
    label: String(row.label ?? `Episode ${row.id ?? fallbackId}`),
    values: row.values ?? {},
    durationS: row.duration_s === undefined ? null : Number(row.duration_s),
    cameras: (row.cameras ?? []).map((camera) => readCamera(camera, base)),
    phone: mediaUrl(row.phone, base),
    rrd: mediaUrl(row.rrd, base),
    href: mediaUrl(row.href, base),
  };
}

function readAbout(about) {
  return (about ?? []).map((fact) => ({ key: String(fact.key), label: String(fact.label), value: fact.value }));
}

/** A manifest, read from `base`: its episodes are the rows, between an index and a duration column. */
export function readManifest(manifest, base) {
  const rows = manifest.episodes.map((episode) =>
    readRow(
      {
        id: episode.index,
        values: { ...episode.values, [INDEX_KEY]: episode.index, [DURATION_KEY]: episode.duration_s },
        duration_s: episode.duration_s,
        cameras: episode.videos,
        phone: episode.phone,
        rrd: episode.rrd,
      },
      base,
    ),
  );
  const columns = [
    { key: INDEX_KEY, label: '#', type: ColumnType.NUMBER },
    ...manifest.columns,
    { key: DURATION_KEY, label: 'Duration', type: ColumnType.NUMBER, decimals: 1, unit: ' s' },
  ].map(readColumn);
  return { title: manifest.title ?? null, about: readAbout(manifest.about), columns: withOptions(columns, rows), rows };
}

/** The host's own rows, read from `base`. A row with no `id` takes its position. */
export function readInline(data, base) {
  const rows = data.rows.map((row, position) => readRow(row, base, position));
  const columns = data.columns.map(readColumn);
  return { title: data.title ?? null, about: readAbout(data.about), columns: withOptions(columns, rows), rows };
}
