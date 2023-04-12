import React, { lazy, ReactNode, Suspense } from 'react'

import type { RouteObject } from 'react-router-dom'

import Loading from '@/components/Loading'
import AppLayout from '@/layouts/AppLayout'

const Home = lazy(() => import('@/pages/Home'))

const List = lazy(() => import('@/pages/List'))

const Form = lazy(() => import('@/pages/Form'))

const Detail = lazy(() => import('@/pages/Detail'))

const Page403 = lazy(() => import('@/pages/403'))

const Page404 = lazy(() => import('@/pages/404'))

const ActiveLearning = lazy(() => import('@/pages/ActiveLearning'))

const Hyperloop = lazy(() => import('@/pages/Hyperloop'))

const Project = lazy(() => import('@/pages/Project'))

const ProjectDetail = lazy(() => import('@/pages/Project/Detail'))

const MLflow = lazy(() => import('@/pages/MLflow'))

const lazyLoad = (children: ReactNode): ReactNode => {
  return <Suspense fallback={<Loading />}>{children}</Suspense>
}

export const routers: RouteObject[] = [
  {
    path: '/',
    element: <AppLayout />,
    children: [
      // {
      //   index: true,
      //   path: '/home',
      //   element: lazyLoad(<Home />)
      // },
      // {
      //   path: '/list',
      //   element: lazyLoad(<List />)
      // },
      // {
      //   path: '/form',
      //   element: lazyLoad(<Form />)
      // },
      // {
      //   path: '/detail',
      //   element: lazyLoad(<Detail />)
      // },
      // {
      //   path: '/403',
      //   element: lazyLoad(<Page403 />)
      // },
      // {
      //   path: '/404',
      //   element: lazyLoad(<Page404 />)
      // },
      {
        path: '/project',
        element: lazyLoad(<ProjectDetail />)
      },
      {
        path: '/project/:name',
        element: lazyLoad(<ProjectDetail />)
      },
      {
        index: true,
        path: '/activeLearning',
        element: lazyLoad(<ActiveLearning />)
      },
      {
        path: '/activeLearning/job/:id',
        element: lazyLoad(<Hyperloop />)
      },
      {
        path: '/mlflow',
        element: lazyLoad(<MLflow />)
      },
      {
        path: '*',
        element: lazyLoad(<Page404 />)
      }
    ]
  }
]
