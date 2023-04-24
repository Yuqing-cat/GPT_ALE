import { Modal } from 'antd'
import { action, makeObservable, observable, toJS, runInAction } from 'mobx'
import type { NavigateFunction } from 'react-router-dom'

import API from '@/api'
import { JobProgress } from '@/api/gpt_ale/interface'
// import config from '@/config'

class GlobalStore {
  jobsCount = 0
  jobs: JobProgress[] = []
  downJobs: any = {}
  isShowModal = false
  navigate?: NavigateFunction

  constructor() {
    makeObservable(this, {
      jobsCount: observable,
      jobs: observable,
      getJobsProgress: action.bound,
      setJobsCount: action.bound,
      showJobsDoneModal: action.bound,
      growingProgress: action.bound
    })
    // this.growingProgress()
  }

  growingProgress() {
    // const isUpdate = false
    const newJobs = toJS(this.jobs)
    runInAction(() => {
      this.jobs = newJobs
      this.showJobsDoneModal()
    })
  }

  async getJobsProgress() {
    try {
      const result = (await API.gpt_ale.getJobsProgress()) || []
      runInAction(() => {
        this.jobs = result
        this.setJobsCount(result.length)
        this.showJobsDoneModal()
      })
    } catch {
      //
    }
  }

  showJobsDoneModal = async () => {
    const job = this.jobs.find((item) => item.progress === 100)
    let secondsToGo = 5

    if (job && !this.isShowModal) {
      this.isShowModal = true
      job.flag = true
      const result = await API.gpt_ale.allRuns()
      const jobid = result[result.length - 1].jobid
      if (!this.downJobs[jobid] && !jobid.includes('Running')) {
        this.downJobs[jobid] = true

        const navigateTojob = () => {
          this.navigate?.(`/activeLearning/job/${jobid}`)
        }

        const modal = Modal.confirm({
          title: 'One job has already been done.',
          content: `Automatically jump after ${secondsToGo} seconds.`,
          okText: 'Jump Now',
          onOk: () => {
            this.isShowModal = false
            navigateTojob()
          },
          onCancel: () => {
            clearInterval(timer)
            clearTimeout(timer2)
            modal.destroy()
          }
        })

        const timer = setInterval(() => {
          secondsToGo -= 1
          modal.update({
            content: `Automatically jump after ${secondsToGo} seconds.`
          })
        }, 1000)

        const timer2 = setTimeout(() => {
          this.isShowModal = false
          clearInterval(timer)
          modal.destroy()
          navigateTojob()
        }, secondsToGo * 1000)
      }
    }
  }

  setJobsCount(count: number, flag?: boolean) {
    this.jobsCount = count || 0
  }
}

export default new GlobalStore()
