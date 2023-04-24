import React, { useRef } from 'react'

import { Row, Col, Form, Input, Select, Button } from 'antd'

export interface SearchValue {
  project?: string
  keyword?: string
}

export interface SearchBarProps {
  defaultValues?: SearchValue
  onSearch?: (values: SearchValue) => void
}

const { Item } = Form

const SearchBar = (props: SearchBarProps) => {
  const [form] = Form.useForm()

  const { defaultValues, onSearch } = props

  const timeRef = useRef<any>(null)

  const onChangeKeyword = () => {
    clearTimeout(timeRef.current)
    timeRef.current = setTimeout(() => {
      form.submit()
    }, 350)
  }

  return (
    <Row justify="space-between" style={{ marginBottom: 20 }}>
      <Col>
        <Form form={form} initialValues={defaultValues} layout="inline" onFinish={onSearch}>
          <Item label="Select Project" name="project">
            <Select
              options={[
                {
                  value: '310000',
                  label: 'Project 310000'
                },
                {
                  value: '315000',
                  label: 'Project 315000'
                },
                {
                  value: '325000',
                  label: 'Project 325000'
                }
              ]}
              notFoundContent="No projects found from server"
              placeholder="Project Name"
              style={{ width: 300 }}
            />
          </Item>
          <Item name="keyword">
            <Input placeholder="keyword" onChange={onChangeKeyword} />
          </Item>
        </Form>
      </Col>
      <Col></Col>
    </Row>
  )
}

export default SearchBar
