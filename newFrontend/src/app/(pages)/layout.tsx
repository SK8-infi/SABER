'use client'

// React Imports
import { Suspense } from 'react'
import type { ReactNode } from 'react'

// Component Imports
import Header from '@/components/layout/Header'
import Sidebar from '@/components/layout/Sidebar'
import { SidebarInset } from '@/components/ui/sidebar'
import { Toaster } from '@/components/ui/sonner'
import { RetrievalParamsProvider } from '@/contexts/retrieval-params-context'

const PagesLayout = ({ children }: Readonly<{ children: ReactNode }>) => {
  return (
    <RetrievalParamsProvider>
      <div className='flex h-full w-full min-w-0'>
        <Suspense>
          <Sidebar />
        </Suspense>
        <SidebarInset className='flex flex-1 flex-col'>
          <Header />
          <main className='mx-auto size-full max-w-360 flex-1 px-4 py-6 sm:px-6'>{children}</main>
          <Toaster />
        </SidebarInset>
      </div>
    </RetrievalParamsProvider>
  )
}

export default PagesLayout
