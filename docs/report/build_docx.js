const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  LevelFormat, PageBreak, Footer, PageNumber,
} = require('docx');
const fs = require('fs');

const SONG = { ascii: 'Times New Roman', eastAsia: 'SimSun', hAnsi: 'SimSun' };
const HEI  = { ascii: 'Arial', eastAsia: 'SimHei', hAnsi: 'SimHei' };
const W = 9360;
const R = AlignmentType.RIGHT, C = AlignmentType.CENTER, L = AlignmentType.LEFT;

function runs(s, extra) {
  return s.split('**').map((part, i) =>
    new TextRun(Object.assign({ text: part, bold: i % 2 === 1, font: SONG }, extra || {})));
}
function P(s, firstLine) {
  return new Paragraph({
    children: runs(s),
    spacing: { line: 340, before: 70, after: 70 },
    indent: firstLine === false ? undefined : { firstLine: 420 },
  });
}
function H(s, level) {
  const sizes = { 1: 30, 2: 24, 3: 21 };
  return new Paragraph({
    heading: level === 1 ? HeadingLevel.HEADING_1
      : level === 2 ? HeadingLevel.HEADING_2 : HeadingLevel.HEADING_3,
    spacing: { before: level === 1 ? 360 : 260, after: 140 },
    children: [new TextRun({ text: s, bold: true, font: HEI, size: sizes[level], color: '000000' })],
  });
}
function LI(s) {
  return new Paragraph({
    numbering: { reference: 'dash', level: 0 },
    spacing: { line: 340, before: 50, after: 50 },
    children: runs(s),
  });
}
function CAP(s) {
  return new Paragraph({
    spacing: { before: 200, after: 90 },
    keepNext: true,
    children: [new TextRun({ text: s, bold: true, font: SONG, size: 19 })],
  });
}
function GAP(h) {
  return new Paragraph({ spacing: { after: h || 120 }, children: [new TextRun('')] });
}
function cell(text, w, bold, shade, align) {
  return new TableCell({
    width: { size: w, type: WidthType.DXA },
    shading: shade ? { type: ShadingType.CLEAR, fill: shade, color: 'auto' } : undefined,
    margins: { top: 70, bottom: 70, left: 100, right: 100 },
    children: [new Paragraph({
      alignment: align,
      spacing: { line: 280 },
      children: runs(text, { size: 19 }).map((r) => r),
    })],
  });
}
function T(widths, rows, aligns) {
  const total = widths.reduce((a, b) => a + b, 0);
  const sc = widths.map((x) => Math.round((x / total) * W));
  sc[sc.length - 1] = W - sc.slice(0, -1).reduce((a, b) => a + b, 0);
  return new Table({
    columnWidths: sc,
    width: { size: W, type: WidthType.DXA },
    borders: {
      top:    { style: BorderStyle.SINGLE, size: 8, color: '7F7F7F' },
      bottom: { style: BorderStyle.SINGLE, size: 8, color: '7F7F7F' },
      left:   { style: BorderStyle.SINGLE, size: 4, color: 'BFBFBF' },
      right:  { style: BorderStyle.SINGLE, size: 4, color: 'BFBFBF' },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 4, color: 'BFBFBF' },
      insideVertical:   { style: BorderStyle.SINGLE, size: 4, color: 'BFBFBF' },
    },
    rows: rows.map((r, ri) => new TableRow({
      tableHeader: ri === 0,
      cantSplit: true,
      children: r.map((c, ci) => cell(
        c, sc[ci], ri === 0, ri === 0 ? 'EDEDED' : null,
        ri === 0 ? C : (aligns && aligns[ci] ? aligns[ci] : L))),
    })),
  });
}

// ---- 内容文件的解析：每行一条指令 ----
const ALIGN = { l: L, c: C, r: R };
const lines = fs.readFileSync(__dirname + '/中期报告-内容.txt', 'utf8').split('\n');
const body = [];
let i = 0;
while (i < lines.length) {
  const raw = lines[i]; i++;
  const line = raw.replace(/\s+$/, '');
  if (!line) continue;
  const sp = line.indexOf(' ');
  const tag = sp < 0 ? line : line.slice(0, sp);
  const rest = sp < 0 ? '' : line.slice(sp + 1);
  if (tag === 'TITLE') {
    body.push(new Paragraph({
      alignment: C, spacing: { before: 900, after: 140 },
      children: [new TextRun({ text: rest, bold: true, font: HEI, size: 44 })],
    }));
  } else if (tag === 'SUB') {
    body.push(new Paragraph({
      alignment: C, spacing: { after: 700 },
      children: [new TextRun({ text: rest, font: HEI, size: 24, color: '404040' })],
    }));
  } else if (tag === 'H1' || tag === 'H2' || tag === 'H3') {
    body.push(H(rest, Number(tag[1])));
  } else if (tag === 'P') {
    body.push(P(rest));
  } else if (tag === 'PN') {
    body.push(P(rest, false));
  } else if (tag === 'L') {
    body.push(LI(rest));
  } else if (tag === 'CAP') {
    body.push(CAP(rest));
  } else if (tag === 'GAP') {
    body.push(GAP(Number(rest) || 120));
  } else if (tag === 'PB') {
    body.push(new Paragraph({ children: [new PageBreak()] }));
  } else if (tag === 'T') {
    // T <w1|w2|...> [<a1|a2|...>] 然后连续的 R 行
    const parts = rest.split('  ');
    const widths = parts[0].split('|').map(Number);
    const aligns = parts[1] ? parts[1].split('|').map((x) => ALIGN[x] || L) : null;
    const rows = [];
    while (i < lines.length && lines[i].startsWith('R ')) {
      rows.push(lines[i].slice(2).replace(/\s+$/, '').split('|'));
      i++;
    }
    body.push(T(widths, rows, aligns));
  } else {
    throw new Error('unknown tag: ' + tag + ' @ line ' + i);
  }
}

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: SONG, size: 21 }, paragraph: { spacing: { line: 340 } } },
    },
  },
  numbering: {
    config: [{
      reference: 'dash',
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: '—', alignment: L,
        style: { paragraph: { indent: { left: 480, hanging: 240 } } },
      }],
    }],
  },
  sections: [{
    properties: { page: { margin: { top: 1440, right: 1350, bottom: 1440, left: 1350 } } },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: C,
          children: [new TextRun({ children: [PageNumber.CURRENT], font: SONG, size: 18 })],
        })],
      }),
    },
    children: body,
  }],
});

Packer.toBuffer(doc).then((b) => {
  fs.writeFileSync(process.argv[2] || 'out.docx', b);
  console.log('written', (b.length / 1024).toFixed(0), 'KB;', body.length, 'blocks');
});
