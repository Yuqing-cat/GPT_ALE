// import { TableProps } from 'antd'
import type { TableColumnType, TableProps } from 'antd'
// import type { ColumnType, TableProps } from 'antd/es/table'
import { ResizeHandle, ResizableProps } from 'react-resizable'

export interface ResizeTableProps<T> extends Omit<TableProps<T>, 'columns'> {
  columns?: ResizeColumnType<T>[]
}

// eslint-disable-next-line dot-notation
export interface ResizeColumnType<T> extends TableColumnType<T> {
  resize?: boolean
  minWidth?: number
}

export interface ResizableTitleProps {
  onResize?: ResizableProps['onResize']
  width?: ResizableProps['width']
  minWidth?: number
}

export interface ResizeHandleProps {
  handleAxis?: ResizeHandle
}
