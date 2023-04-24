export type Point = number[]

export interface ChoicePoint {
  id: number
  text: string
  title: string
  label: string
  point: number[]
  score: number
  newPoint?: number[]
  newLabel?: string
  gpt?: string
  ann_by?: string
}

export interface Category {
  value: number
  name: string
  itemStyle: { color: string }
}

export interface Detail {
  heat_map?: string // Record<string, Point[]>
  pie_chart?: string // Record<string, Point[]>
  target_to_label: Record<string, string>
  more_info?: {
    gray_points: ChoicePoint[]
    transformations: {
      index: number
      new_loc: number[]
      old_loc: number[]
    }[]
  }
  all_heat_maps: Record<string, string>
  can_annotate: boolean
  annotations: Record<string, number>
}

export interface AllRunsRespose {
  jobid: string
  number_of_category: number
  accuracy: string
  can_annotate: boolean
  creation_time: string
}

export interface JobProgress {
  jobid: string
  progress: number
  newProgress?: number
  flag?: boolean
}
