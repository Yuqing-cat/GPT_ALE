import React, { useEffect, useState } from 'react'

import Joyride from 'react-joyride'
import type { Step, CallBackProps } from 'react-joyride'

import Tooltip from './Tooltip'
export interface GuideProps {
  show?: boolean
  setPointPosition?: () => void
}

const Guide = (props: GuideProps) => {
  const { show, setPointPosition } = props

  const [notShow, setNotShow] = useState(false)

  const [guideSteps, setGuideSteps] = useState<Step[]>([
    {
      content:
        'This point cloud shows a visualization of the distribution of data points. Different color stands for different categories, while the brightness of the color represents the confidence. Click any point , or use the bulk tool on the top right corner to start',
      placement: 'right',
      target: '#scatterChartCard'
    },
    {
      content:
        'Use this dialog box to annotate a single data point. Pick any existing categories in the list, or input your own category on the bottom before confirming. Suggestion from GPT-3 has been loaded, click the button on the bottom right to directly accept GPT-3 suggestion. Once confirmed, next point will be prompted. ',
      target: '#point'
    },
    {
      content:
        'Use this menu to perform bulk annotation. This will only be enabled when selected a region with bulk tool on the point cloud. ',
      placement: 'bottom',
      target: '#selectedPointsBtn'
    },
    {
      content:
        'This is a list of recommended annotation targets, these samples are taken as the most valuable ones, but this is not enforced, feel free to annotate any data point as you wish. ',
      placement: 'bottom',
      target: '#drawerListBtn'
    },
    {
      content:
        'These are charts that show the overall annotation progress, categorical distribution, and the model prediction from last job. ',
      placement: 'left',
      target: '#rightCharts'
    },
    {
      content:
        'Once annotated enough data points, you can either continue annotating more points, or submit annotation and trigger next round of hyperloop. After the hyperloop is triggered, expect 3-5 minutes before you can see next annotation job in the list. ',
      placement: 'bottom',
      target: '#startHyperloopBtn'
    }
  ])

  const callback = (data: CallBackProps) => {
    const { index } = data
    if (index === 0) {
      setPointPosition?.()
    }
  }

  useEffect(() => {
    if (show) {
      setNotShow(localStorage.getItem('not-show-guide') === '1')
    }
  }, [show])

  return show && !notShow ? (
    <Joyride
      continuous
      showProgress
      showSkipButton
      scrollOffset={60}
      run={true}
      steps={guideSteps}
      tooltipComponent={Tooltip}
      callback={callback}
    />
  ) : null
}

export default Guide
