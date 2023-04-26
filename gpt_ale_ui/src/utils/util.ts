const opt = Object.prototype.toString

export function isArray(obj: any): obj is any[] {
  return opt.call(obj) === '[object Array]'
}

export function isObject(obj: any): obj is { [key: string]: any } {
  return opt.call(obj) === '[object Object]'
}

// eslint-disable-next-line @typescript-eslint/no-empty-function
const NOOP = () => {}

export const isServerRendering = (() => {
  try {
    return !(typeof window !== 'undefined' && document !== undefined)
  } catch (e) {
    return true
  }
})()

export const on = (() => {
  if (isServerRendering) {
    return NOOP
  }
  return function (
    element: any,
    event: string,
    handler: EventListener | EventListenerObject | (() => void),
    options?: boolean | AddEventListenerOptions
  ) {
    element && element.addEventListener(event, handler, options || false)
  }
})()

export const off = (() => {
  if (isServerRendering) {
    return NOOP
  }
  return function (
    element: any,
    event: string,
    handler: EventListener | EventListenerObject | (() => void),
    options?: boolean | AddEventListenerOptions
  ) {
    element && element.removeEventListener(event, handler, options || false)
  }
})()
