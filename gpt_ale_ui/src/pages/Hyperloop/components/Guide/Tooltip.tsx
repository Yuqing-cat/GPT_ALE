import React from 'react'

import { Card, Typography, Button, Checkbox, Space } from 'antd'
import { CheckboxChangeEvent } from 'antd/lib/checkbox'
import type { TooltipRenderProps } from 'react-joyride'

const { Title, Paragraph } = Typography

const Tooltip = ({
  backProps,
  continuous,
  index,
  isLastStep,
  primaryProps,
  skipProps,
  step,
  tooltipProps
}: TooltipRenderProps) => {
  const onChange = (e: CheckboxChangeEvent) => {
    localStorage.setItem('not-show-guide', e.target.checked ? '1' : '0')
  }

  return (
    <Card
      {...tooltipProps}
      bordered={false}
      style={{ maxWidth: 420, minWidth: 290, overflow: 'hidden' }}
    >
      <div>
        <Title level={3}>{step.title}</Title>
        <Paragraph>{step.content}</Paragraph>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Checkbox onChange={onChange}>Don't show this again</Checkbox>
        <Space>
          {!isLastStep && (
            <Button {...skipProps} type="link">
              skip
            </Button>
          )}
          {index > 0 && (
            <Button {...backProps} type="default">
              Back
            </Button>
          )}
          <Button {...primaryProps} type="primary">
            {continuous ? 'Next' : 'Close'}
          </Button>
        </Space>
      </div>
    </Card>
  )
}
export default Tooltip
