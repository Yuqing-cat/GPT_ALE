import ColorHash from 'color-hash'
import { number } from 'echarts'

const colorHash = new ColorHash()

export const getHashCode = (str = '') => {
  let hash = 1315423911
  for (let i = str.length - 1; i >= 0; i--) {
    const ch = str.charCodeAt(i)
    hash ^= (hash << 5) + ch + (hash >> 2)
  }
  return hash & 0x7ffffff
}

const presetColorValue = (value: number, score: number) => {
  const x = 1 - score
  return Math.floor(value * score + 128 * x)
}

export const getColor = (value = '', score?: number) => {
  if (score !== undefined) {
    const [r, g, b] = colorHash.rgb(value)
    return `#${presetColorValue(r, score).toString(16)}${presetColorValue(g, score).toString(
      16
    )}${presetColorValue(b, score).toString(16)}`
  } else {
    return colorHash.hex(value)
  }

  // return colorHash.rgb().hex(value) //`#${getHashCode(value).toString(16).substring(0, 6)}`
}
