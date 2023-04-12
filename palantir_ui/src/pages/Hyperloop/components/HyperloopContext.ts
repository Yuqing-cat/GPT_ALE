import { createContext } from 'react'
import type { MutableRefObject } from 'react'

import { Category, ChoicePoint, Point } from '@/api/palantir/interface'

export const HyperloopContext = createContext<{
  loading?: boolean
  points?: MutableRefObject<ChoicePoint[] | Record<string, Point[]>>
  choicePoints?: MutableRefObject<ChoicePoint[]>
  updateData?: MutableRefObject<Record<number, string>>
  categories?: Category[]
  categorieNames?: MutableRefObject<string[]>
  occurance?: MutableRefObject<Record<string, Point[]>>
  transformations?: MutableRefObject<
    {
      index: number
      new_loc: number[]
      old_loc: number[]
    }[]
  >
  isFianl?: boolean
  isStartHyperloop?: MutableRefObject<boolean>
  mapGPT?: MutableRefObject<Record<number, string>>
  onCreateCategory?: (names: string[]) => void
  onChangeCategory?: (data: ChoicePoint, name: string) => void
  onBrushselected?: (params: any, option: any) => void
  onBrushEnd?: (params: any) => void
}>({})
