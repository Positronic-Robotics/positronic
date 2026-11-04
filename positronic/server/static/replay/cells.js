// The content of one cell of the episode list, by the type of its column.
import { cellText, ColumnType } from './data.js';

export function element(tag, attributes = {}, children = []) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (value !== null && value !== undefined) node.setAttribute(name, value);
  }
  node.append(...children);
  return node;
}

/** A colour's relative luminance, from a `#rgb` or `#rrggbb` colour; null for any other spelling. */
function luminance(color) {
  const hex = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(color)?.[1];
  if (!hex) return null;
  const full = hex.length === 3 ? [...hex].map((digit) => digit + digit).join('') : hex;
  const [r, g, b] = [0, 2, 4].map((at) => {
    const channel = parseInt(full.slice(at, at + 2), 16) / 255;
    return channel <= 0.03928 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** The text colour that reads on a badge of `color`. */
export function inkOn(color) {
  const light = luminance(color);
  return light !== null && light < 0.4 ? '#ffffff' : '#111111';
}

/** A number cell's text: its value at the column's decimals, with its unit; '-' where the value is empty. */
function numberText(column, value) {
  const number = value !== null && typeof value === 'object' ? value.text : value;
  if (number === null || number === undefined || number === '') return '-';
  if (typeof number !== 'number') return String(number);
  return `${column.decimals === null ? String(number) : number.toFixed(column.decimals)}${column.unit}`;
}

/** The nodes one cell holds. `link` wraps the text of a text cell, for the row's own page. */
export function cellContent(column, value, link = null) {
  // A null or absent value is an empty cell of any type.
  const fields = value !== null && typeof value === 'object' ? value : {};
  const title = fields.title;
  if (column.type === ColumnType.NUMBER) return [numberText(column, value)];
  const text = cellText(value);
  if (column.type === ColumnType.BADGE) {
    if (!text) return ['-'];
    const color = fields.color;
    if (!color) return [element('span', { part: 'badge', title }, [text])];
    const ink = fields.ink ?? inkOn(color);
    return [element('span', { part: 'badge', title, style: `background: ${color}; color: ${ink}` }, [text])];
  }
  if (column.type === ColumnType.ICON) {
    const icon = fields.icon;
    const image = icon ? [element('img', { part: 'icon', src: icon, alt: '' })] : [];
    return [element('span', { part: 'with-icon', title }, [...image, text])];
  }
  return link ? [element('a', { part: 'row-link', href: link, title }, [text])] : [text];
}
