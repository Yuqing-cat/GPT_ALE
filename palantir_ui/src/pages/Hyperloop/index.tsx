import React, { useState, useEffect, useRef, useCallback } from 'react'

import { ExclamationCircleOutlined, UnorderedListOutlined } from '@ant-design/icons'
import { Row, Col, Card, Button, Modal, message, Descriptions, Space } from 'antd'
import { useParams } from 'react-router'

import API from '@/api'
import { Category, ChoicePoint, Point } from '@/api/palantir/interface'
import PagePanel from '@/components/PagePanel'
import { observer, useStore } from '@/hooks'
import { getColor } from '@/utils/color'

import DrawerPoints, { DrawerPointsHandle } from './components/DrawerPoints'
import DrawerSelectPoints, { DrawerSelectPointsHandle } from './components/DrawerSelectPoints'
import Guide from './components/Guide'
import HeatmapChart from './components/HeatmapChart'
import { HyperloopContext } from './components/HyperloopContext'
import PerChart from './components/PerChart'
import ScatterChart, { ScatterChartHandle } from './components/ScatterChart'

import styles from './index.module.less'

const { Item } = Descriptions

const Hyperloop = () => {
  const { globalStore } = useStore()
  const { getJobsProgress } = globalStore
  const { id } = useParams()

  const [loading, setLoading] = useState<boolean>(true)
  const [categories, setCategories] = useState<Category[]>([])
  const categorieNamesRef = useRef<string[]>([])
  const scatterRef = useRef<ScatterChartHandle>(null)
  const drawerPointsRef = useRef<DrawerPointsHandle>(null)
  const drawerSelectPointsRef = useRef<DrawerSelectPointsHandle>(null)
  const selectdPointsRef = useRef<ChoicePoint[]>([])

  const pieValuesRef = useRef<number[][]>([])
  const pointsRef = useRef<ChoicePoint[] | Record<string, Point[]>>([])
  const choicePointsRef = useRef<ChoicePoint[]>([])
  const updateDataRef = useRef<Record<number, string>>({})
  const occuranceRef = useRef<Record<string, Point[]>>({})
  const mapGPTRef = useRef<Record<number, string>>({})

  const transformationsRef = useRef<
    {
      index: number
      new_loc: number[]
      old_loc: number[]
    }[]
  >([])
  const isStartHyperloopRef = useRef<boolean>(false)
  const [isFianl, setIsFianl] = useState<boolean>(false)
  const [showStartBar, setShowStartBar] = useState<boolean>(false)

  const [disabledButton, setDisabledButton] = useState<boolean>(true)

  const choicePointsNextRef = useRef(new Set())

  const stepPointRef = useRef<HTMLDivElement>(null)

  const fetchData = useCallback(async () => {
    try {
      const detail = await API.palantir.runStatus(id!)

      occuranceRef.current = Object.keys(detail.all_heat_maps || {}).reduce(
        (obj: any, key: string) => {
          obj[key] = JSON.parse(detail.all_heat_maps[key] || '[]')
          return obj
        },
        {} as any
      )
      pieValuesRef.current = JSON.parse(detail.pie_chart || '[]') as number[][]
      categorieNamesRef.current = Object.values(detail.target_to_label).reduce(
        (values: string[], name: string) => {
          if (name !== 'unknown') {
            values.push(name)
          }
          return values
        },
        []
      )
      const categorieSet = new Set(categorieNamesRef.current)
      const categories = categorieNamesRef.current.map((name, index) => {
        const [, value] = pieValuesRef.current[index] || [0, 0]

        return {
          value: value || detail.annotations[name] || 0,
          name: name,
          itemStyle: {
            color: getColor(name)
          }
        }
      })

      if (detail.can_annotate) {
        const choicePoints = await API.palantir.annotationTarget()
        updateDataRef.current = JSON.parse(localStorage.getItem(`job-${id}`) || '{}')
        pointsRef.current = detail.more_info?.gray_points || []
        transformationsRef.current = detail.more_info?.transformations || []
        const transformationsKey = transformationsRef.current.reduce((keys: any, item) => {
          keys[item.index] = item
          return keys
        }, {} as any)
        const transformPoints2: ChoicePoint[] = []
        const staticPoint2: ChoicePoint[] = []
        choicePoints.map((item) => {
          const choice = transformationsKey[item.id]
          if (choice) {
            transformPoints2.push(item)
          } else {
            staticPoint2.push(item)
          }
        })

        choicePointsRef.current = transformPoints2.concat(staticPoint2)

        pointsRef.current.forEach((item) => {
          const { id, gpt } = item
          if (gpt && gpt !== 'unknown') {
            mapGPTRef.current[id] = gpt
          }

          const newLabel = updateDataRef.current[id]
          if (newLabel) {
            if (!categorieSet.has(newLabel)) {
              categorieSet.add(newLabel)
              categories.push({
                value: 0,
                name: newLabel,
                itemStyle: {
                  color: getColor(newLabel)
                }
              })
            }

            const category = categories.find((item) => item.name === newLabel)
            category!.value++
          }
        })
        choicePointsRef.current.forEach((item, index) => {
          if (updateDataRef.current[item.id]) {
            choicePointsNextRef.current.add(index)
          }
        })
        setCategories(categories)
        setShowStartBar(true)
      } else {
        const allPoint = await API.palantir.jobPointCloud(id!)
        localStorage.removeItem(`job-${id}`)
        setCategories(categories)
        if (allPoint.category) {
          pointsRef.current = allPoint.category || {}
          setShowStartBar(true)
        } else {
          pointsRef.current = {}
        }

        setIsFianl(true)
      }
    } catch (e: any) {
      message.error(e.message)
      setIsFianl(true)
    } finally {
      setLoading(false)
    }
  }, [id])

  const onStartHyperloop = () => {
    Modal.confirm({
      title: 'Confirm Start Hyperloop',
      icon: <ExclamationCircleOutlined />,
      // content: 'Start Hyperloop',
      onOk: async () => {
        try {
          await API.palantir.updateAnnotation(
            id!,
            Object.keys(updateDataRef.current).reduce((list: any[], pid: string) => {
              if (pid) {
                const id = +pid
                const label = updateDataRef.current?.[id]
                list.push({
                  id,
                  label,
                  ann_by: mapGPTRef.current[id] === label ? 'GPT' : 'SME'
                })
              }
              return list
            }, [] as any[])
          )
          message.success('Start Hyperloop success.')
          localStorage.removeItem(`job-${id}`)
          isStartHyperloopRef.current = true
          setShowStartBar(false)
          getJobsProgress()
        } catch {
          message.error('Start Hyperloop failure!')
        }
      }
    })
  }

  const onCreateCategory = (names: string[]) => {
    const newList: Category[] = []
    setCategories((state) => {
      names.forEach((name) => {
        if (state.findIndex((item) => item.name === name) === -1) {
          newList.push({
            value: 0,
            name,
            itemStyle: { color: getColor(name) }
          })
        }
      })
      return newList.concat(state)
    })
  }

  const showNextTooltip = (currentInex: number) => {
    if (currentInex !== -1) {
      choicePointsNextRef.current.add(currentInex)
    }

    let nextIndex = currentInex
    let pointIndex = -1
    do {
      if (choicePointsNextRef.current.size === choicePointsRef.current.length) {
        pointIndex === -1
        break
      } else if (choicePointsRef.current.length > nextIndex) {
        if (nextIndex !== -1 && !choicePointsNextRef.current.has(nextIndex)) {
          pointIndex = nextIndex
        } else {
          nextIndex++
        }
      } else {
        nextIndex = 0
      }
    } while (pointIndex === -1)

    if (pointIndex > -1) {
      scatterRef.current?.showTooltip(pointIndex)
    }
  }

  const onChangeCategory = (data: ChoicePoint, name: string) => {
    const { id: pid } = data
    const choicePoints = pointsRef.current as ChoicePoint[]
    const index = choicePoints.findIndex((item) => item.id === pid)
    const currentIndex = choicePointsRef.current.findIndex((item) => item.id === pid)

    if (index > -1) {
      const oleName = updateDataRef.current[pid]
      updateDataRef.current[pid] = name
      drawerPointsRef.current?.forceUpdate()
      showNextTooltip(currentIndex)

      if (oleName !== name) {
        setCategories((state) => {
          state.forEach((item) => {
            if (item.name === name) {
              item.value++
            } else if (item.name === oleName) {
              item.value--
            }
          })
          return [...state]
        })
      }
      localStorage.setItem(`job-${id}`, JSON.stringify(updateDataRef.current))
    }
  }

  const onPublishModel = () => {
    Modal.confirm({
      title: 'Comfirm the model publishing?',
      icon: <ExclamationCircleOutlined />
      // content: 'Start Hyperloop',
    })
  }

  const onShowDrawer = () => {
    drawerPointsRef.current?.open()
  }

  const onDrawerEdit = (point: ChoicePoint, name: string) => {
    const { id: pid } = point

    localStorage.setItem(`job-${id}`, JSON.stringify(updateDataRef.current))
    scatterRef.current?.unBrush()
    drawerPointsRef.current?.forceUpdate()
    drawerSelectPointsRef.current?.close()
    setCategories((state) => {
      const oleName = updateDataRef.current[pid]
      updateDataRef.current[pid] = name
      if (oleName !== name) {
        state.forEach((item) => {
          if (item.name === name) {
            item.value++
          } else if (item.name === oleName) {
            item.value--
          }
        })
      }
      return [...state]
    })
    setDisabledButton(true)
  }

  const onBrushselected = (params: any, option: any) => {
    const { batch } = params

    const indexSet = new Set<number>()
    selectdPointsRef.current = []

    batch.forEach((item: any) => {
      item.selected.forEach((selected: any) => {
        const { seriesIndex, dataIndex } = selected
        const serie = option.series[seriesIndex]
        const data = serie.data

        dataIndex.forEach((index: number) => {
          const point = { ...data[index][2] }
          point.newLabel = updateDataRef.current[point.id] || point.label
          if (!indexSet.has(point.id)) {
            indexSet.add(point.id)
            selectdPointsRef.current.push(point)
          }
        })
      })
    })
  }

  const onBrushEnd = () => {
    setDisabledButton(selectdPointsRef.current.length === 0)
  }

  const onBatchConfirm = (points: ChoicePoint[]) => {
    const temp = [...categories]
    points.forEach((item) => {
      const { id, newLabel } = item
      if (id !== undefined) {
        const oleName = updateDataRef.current[id]
        updateDataRef.current[id] = newLabel!

        if (oleName !== newLabel) {
          temp.forEach((item) => {
            if (item.name === newLabel) {
              item.value++
            } else if (item.name === oleName) {
              item.value--
            }
          })
        }
      }
    })
    localStorage.setItem(`job-${id}`, JSON.stringify(updateDataRef.current))
    scatterRef.current?.unBrush()
    drawerPointsRef.current?.forceUpdate()
    drawerPointsRef.current?.close()
    drawerSelectPointsRef.current?.close()
    setCategories(temp)
    setDisabledButton(true)
  }

  const onShowDrawerSelectPoints = () => {
    selectdPointsRef.current.sort((a: any, b: any) => {
      return a.score - b.score
    })
    drawerSelectPointsRef.current?.open()
  }

  const setStepPointPosition = (point: number[]) => {
    const [left, top] = point
    stepPointRef.current?.setAttribute('style', `left: ${left - 9}px; top:${top - 9}px;`)
  }

  const onReset = () => {
    updateDataRef.current = {}
    localStorage.removeItem(`job-${id}`)
    drawerPointsRef.current?.update()
    const categories = categorieNamesRef.current.map((name, index) => {
      const [, value] = pieValuesRef.current[index] || [0, 0]
      return {
        value,
        name: name,
        itemStyle: {
          color: getColor(name)
        }
      }
    })
    setCategories(categories)
    scatterRef.current?.unBrush()
  }

  useEffect(() => {
    fetchData()
  }, [id])

  return (
    <HyperloopContext.Provider
      value={{
        loading,
        isFianl,
        points: pointsRef,
        choicePoints: choicePointsRef,
        categories,
        categorieNames: categorieNamesRef,
        occurance: occuranceRef,
        updateData: updateDataRef,
        transformations: transformationsRef,
        isStartHyperloop: isStartHyperloopRef,
        mapGPT: mapGPTRef,
        onCreateCategory,
        onChangeCategory,
        onBrushselected,
        onBrushEnd
      }}
    >
      <PagePanel
        extra={
          <Space id="guide-1">
            {showStartBar && (
              <Button
                id="startHyperloopBtn"
                disabled={Object.keys(updateDataRef.current).length < 5}
                type="primary"
                onClick={onStartHyperloop}
              >
                Start hyperloop
              </Button>
            )}

            {!loading && isFianl && (
              <Button type="primary" onClick={onPublishModel}>
                Publish Model
              </Button>
            )}
            {!loading && !isFianl && (
              <>
                <Button
                  id="selectedPointsBtn"
                  disabled={disabledButton}
                  onClick={onShowDrawerSelectPoints}
                >
                  Selected Points
                </Button>
                <Button
                  disabled={Object.keys(updateDataRef.current).length === 0}
                  type="ghost"
                  onClick={onReset}
                >
                  Reset
                </Button>
                <Button
                  id="drawerListBtn"
                  icon={<UnorderedListOutlined />}
                  onClick={onShowDrawer}
                />
              </>
            )}
          </Space>
        }
        body={
          showStartBar && (
            <Descriptions size="small" column={3}>
              <Item label="Confirmed">{Object.keys(updateDataRef.current).length}</Item>
              <Item label="Remaining">
                {choicePointsRef.current.length - Object.keys(updateDataRef.current).length}
              </Item>
            </Descriptions>
          )
        }
        title={`Job: ${id}`}
      >
        <Row gutter={[20, 20]}>
          <Col
            lg={24}
            md={24}
            sm={24}
            xl={18}
            xs={24}
            style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 200px)' }}
          >
            <Card
              id="scatterChartCard"
              style={{ flex: 1, display: 'flex', flexDirection: 'column' }}
              bodyStyle={{ padding: 0, flex: 1 }}
              bordered={false}
            >
              <ScatterChart
                ref={scatterRef}
                className={styles.scatterChart}
                getPoint={setStepPointPosition}
              />
              <div ref={stepPointRef} id="point" className={styles.stepPoint}></div>
            </Card>
          </Col>
          <Col
            lg={24}
            md={24}
            sm={24}
            xl={6}
            xs={24}
            style={{ display: 'flex', flexDirection: 'column' }}
            id="rightCharts"
          >
            <Card
              id="guide-card-2"
              title="Categories"
              style={{ marginBottom: 20, flex: 1, display: 'flex', flexDirection: 'column' }}
              bodyStyle={{ padding: 0, flex: 1 }}
            >
              <PerChart />
            </Card>
            <HeatmapChart />
          </Col>
        </Row>
      </PagePanel>
      <DrawerPoints ref={drawerPointsRef} onConfirm={onBatchConfirm} />
      <DrawerSelectPoints
        ref={drawerSelectPointsRef}
        points={selectdPointsRef}
        onConfirm={onBatchConfirm}
      />
      <Guide
        setPointPosition={() => {
          scatterRef.current?.getPoint(0)
        }}
        show={!loading && !isFianl}
      />
    </HyperloopContext.Provider>
  )
}

export default observer(Hyperloop)
