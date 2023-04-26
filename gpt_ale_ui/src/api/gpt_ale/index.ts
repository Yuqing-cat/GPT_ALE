import config from '@/config'
import http from '@/utils/http'

import { AllRunsRespose, ChoicePoint, Detail, JobProgress, Point } from './interface'

const API_PATH = `${config.API_PATH}`

export const allRuns = (): Promise<AllRunsRespose[]> => {
  return http.get(`${API_PATH}/all_runs`)
}

export const allStatus = () => {
  return http.get(`${API_PATH}/all_status`)
}

export const allPoints = (): Promise<Point[]> => {
  return http.get(`${API_PATH}/all_points`)
}

export const annotationTarget = (): Promise<ChoicePoint[]> => {
  return http.get(`${API_PATH}/annotation_target`)
}

export const runStatus = (id: string): Promise<Detail> => {
  return http.get(`${API_PATH}/run_status`, { run_id: id })
}

export const updateAnnotation = (
  id: string,
  playload: { id: number | string; label: string; ann_by?: string }[]
) => {
  return http.post(`${API_PATH}/update_annotation?run_id=${id}`, playload)
}

export const jobPointCloud = (id: string): Promise<{ category: Record<string, Point[]> }> => {
  return http.get(`${API_PATH}/job_point_cloud`, { run_id: id })
}

export const getLineCharts = (name: string) => {
  return http.get(`${API_PATH}/line_charts`, { name })
}

export const getJobsProgress = (): Promise<JobProgress[]> => {
  return http.get(`${API_PATH}/all_unfinished_runs`)
}

export const getConfig = () => {
  return http.get(`${API_PATH}/config`)
}
