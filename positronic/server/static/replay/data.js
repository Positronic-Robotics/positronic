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

const isObject = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
const isEmpty = (value) => value === null || value === undefined;

/** Logs a host input of the wrong shape, and returns the default the element uses in its place. */
export function wrongInput(input, value, fallback, fallbackText) {
  console.error(`${input} has the wrong shape (${JSON.stringify(value)}); the element uses ${fallbackText}`);
  return fallback;
}

/** The entries of `list` that `read(entry, index)` takes; `read` logs an entry it drops and returns null for it. */
function entries(list, read) {
  return list.map(read).filter((entry) => entry !== null);
}

/** Refuses a list whose shape the element cannot read; the element then shows that the replay did not load. */
function refuse(input, value, wanted) {
  throw new TypeError(`${input} is not ${wanted}: ${JSON.stringify(value)}`);
}

/** A media URL: a string, or an object with `src` (relative to `base`) or `url`. Null where the value is empty. */
function mediaUrl(input, value, base) {
  if (isEmpty(value) || value === '') return null;
  const spelled = isObject(value) ? (value.src ?? value.url) : value;
  if (typeof spelled !== 'string') return wrongInput(input, value, null, 'no media');
  try {
    return new URL(spelled, base).href;
  } catch (error) {
    if (!(error instanceof TypeError)) throw error;
    return wrongInput(input, value, null, 'no media');
  }
}

/** A number, or null where the value is empty. */
function number(input, value) {
  if (isEmpty(value)) return null;
  return Number.isFinite(value) ? value : wrongInput(input, value, null, 'no number');
}

function readCamera(input, camera, base) {
  if (!isObject(camera) || isEmpty(camera.camera)) return wrongInput(input, camera, null, 'no camera');
  const src = mediaUrl(input, camera, base);
  if (src === null) return null;
  return {
    camera: String(camera.camera),
    label: isEmpty(camera.label) ? null : String(camera.label),
    src,
    width: number(`${input}.width`, camera.width),
    height: number(`${input}.height`, camera.height),
    durationS: number(`${input}.duration_s`, camera.duration_s) ?? 0,
  };
}

function readFilter(input, filter) {
  if (!filter) return null;
  if (filter !== true && !isObject(filter)) return wrongInput(input, filter, null, 'no filter');
  const spelled = filter === true ? {} : filter;
  if (!isEmpty(spelled.options) && !Array.isArray(spelled.options)) {
    return wrongInput(`${input}.options`, spelled.options, null, 'no filter');
  }
  const readOption = (option, at) => {
    const [value, label] = Array.isArray(option) ? option : isObject(option) ? [option.value, option.label] : [];
    if (isEmpty(value)) return wrongInput(`${input}.options[${at}]`, option, null, 'no option');
    return { value: String(value), label: String(label ?? value) };
  };
  const options = spelled.options && entries(spelled.options, readOption);
  return { all: spelled.all === undefined ? 'All' : String(spelled.all), options: options ?? null };
}

function readColumn(column, at) {
  const input = `columns[${at}]`;
  if (!isObject(column) || isEmpty(column.key)) return wrongInput(input, column, null, 'no column');
  let type = column.type ?? ColumnType.TEXT;
  if (!COLUMN_TYPES.has(type)) type = wrongInput(`${input}.type`, type, ColumnType.TEXT, 'a text column');
  return {
    key: String(column.key),
    label: String(column.label ?? column.key),
    type,
    decimals: number(`${input}.decimals`, column.decimals),
    unit: isEmpty(column.unit) ? '' : String(column.unit),
    hover: isEmpty(column.hover) ? null : String(column.hover),
    filter: readFilter(`${input}.filter`, column.filter),
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

function readRow(row, base, at) {
  const input = `rows[${at}]`;
  if (!isObject(row)) return wrongInput(input, row, null, 'no row');
  const id = String(row.id ?? at);
  const values =
    isEmpty(row.values) || isObject(row.values)
      ? (row.values ?? {})
      : wrongInput(`${input}.values`, row.values, {}, 'no values');
  let cameras = row.cameras ?? [];
  if (!Array.isArray(cameras)) cameras = wrongInput(`${input}.cameras`, cameras, [], 'no cameras');
  return {
    id,
    label: String(row.label ?? `Episode ${id}`),
    values,
    durationS: number(`${input}.duration_s`, row.duration_s),
    cameras: entries(cameras, (camera, each) => readCamera(`${input}.cameras[${each}]`, camera, base)),
    phone: mediaUrl(`${input}.phone`, row.phone, base),
    rrd: mediaUrl(`${input}.rrd`, row.rrd, base),
    href: mediaUrl(`${input}.href`, row.href, base),
  };
}

function readAbout(about) {
  if (isEmpty(about)) return [];
  if (!Array.isArray(about)) return wrongInput('about', about, [], 'no facts');
  return entries(about, (fact, at) => {
    if (!isObject(fact) || isEmpty(fact.key)) return wrongInput(`about[${at}]`, fact, null, 'no fact');
    return { key: String(fact.key), label: String(fact.label ?? fact.key), value: fact.value };
  });
}

/** A manifest, read from `base`: its episodes are the rows, between an index and a duration column. */
export function readManifest(manifest, base) {
  if (!isObject(manifest)) refuse('the manifest', manifest, 'an object');
  if (!Array.isArray(manifest.episodes)) refuse('the manifest episodes', manifest.episodes, 'an array');
  if (!Array.isArray(manifest.columns)) refuse('the manifest columns', manifest.columns, 'an array');
  const rows = entries(manifest.episodes, (episode, at) => {
    if (!isObject(episode)) return wrongInput(`episodes[${at}]`, episode, null, 'no episode');
    const values = isObject(episode.values)
      ? episode.values
      : wrongInput(`episodes[${at}].values`, episode.values, {}, 'no values');
    return readRow(
      {
        id: episode.index,
        values: { ...values, [INDEX_KEY]: episode.index, [DURATION_KEY]: episode.duration_s },
        duration_s: episode.duration_s,
        cameras: episode.videos,
        phone: episode.phone,
        rrd: episode.rrd,
      },
      base,
      at,
    );
  });
  const columns = [
    readColumn({ key: INDEX_KEY, label: '#', type: ColumnType.NUMBER }, '#'),
    ...entries(manifest.columns, readColumn),
    readColumn({ key: DURATION_KEY, label: 'Duration', type: ColumnType.NUMBER, decimals: 1, unit: ' s' }, 'duration'),
  ];
  const title = isEmpty(manifest.title) ? null : String(manifest.title);
  return { title, about: readAbout(manifest.about), columns: withOptions(columns, rows), rows };
}

/** The host's own rows, read from `base`. A row with no `id` takes its position. */
export function readInline(data, base) {
  if (!isObject(data)) refuse('the inline rows', data, 'an object');
  if (!Array.isArray(data.rows)) refuse('the inline rows', data.rows, 'an array');
  if (!Array.isArray(data.columns)) refuse('the inline columns', data.columns, 'an array');
  const rows = entries(data.rows, (row, position) => readRow(row, base, position));
  const columns = entries(data.columns, readColumn);
  const title = isEmpty(data.title) ? null : String(data.title);
  return { title, about: readAbout(data.about), columns: withOptions(columns, rows), rows };
}
