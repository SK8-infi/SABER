'use client'

// Next Imports
import { Fragment } from 'react'

import { usePathname } from 'next/navigation'

// Third-party Imports
import { Zap, Database, Cpu } from 'lucide-react'

// Component Imports
import ModeToggle from '@/components/layout/ModeToggle'
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator
} from '@/components/ui/breadcrumb'
import { Separator } from '@/components/ui/separator'
import { SidebarTrigger } from '@/components/ui/sidebar'

// Context Imports
import { useRetrievalParams } from '@/contexts/retrieval-params-context'

const Header = () => {
  const pathname = usePathname()
  const { telemetry } = useRetrievalParams()

  const segments = pathname.split('/').filter(Boolean)

  return (
    <header className='bg-card sticky top-0 z-50 border-b'>
      <div className='mx-auto flex max-w-360 items-center justify-between gap-6 px-4 py-2 sm:px-6'>
        <div className='flex items-center gap-4'>
          <SidebarTrigger className='[&_svg]:size-5!' />
          <Separator orientation='vertical' className='hidden h-4! data-vertical:self-center sm:block' />
          <Breadcrumb className='hidden sm:block'>
            <BreadcrumbList>
              {segments.map((segment, index) => {
                const isLast = index === segments.length - 1
                const label = segment.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
                const href = '/' + segments.slice(0, index + 1).join('/')

                return (
                  <Fragment key={href}>
                    <BreadcrumbItem>
                      {isLast ? <BreadcrumbPage>{label}</BreadcrumbPage> : <BreadcrumbLink>{label}</BreadcrumbLink>}
                    </BreadcrumbItem>
                    {!isLast && <BreadcrumbSeparator />}
                  </Fragment>
                )
              })}
            </BreadcrumbList>
          </Breadcrumb>
        </div>
        <div className='flex items-center gap-3'>
          <ModeToggle />

          {/* Live Telemetry Pill Cards */}
          <div className='hidden md:flex items-center gap-2.5 ml-2'>
            {/* LATENCY */}
            <div className='flex items-center gap-2.5 px-3 py-1.5 rounded-2xl border border-border/60 bg-background/50 shadow-2xs'>
              <div className='flex items-center justify-center size-7 rounded-xl bg-orange-500/10 text-orange-500 dark:bg-amber-500/15 dark:text-amber-400 shrink-0'>
                <Zap className='size-3.5' />
              </div>
              <div className='flex flex-col text-left leading-tight'>
                <span className='text-[9px] font-bold tracking-wider text-muted-foreground uppercase font-sans'>LATENCY</span>
                <span className='text-xs font-bold text-foreground font-sans'>
                  {telemetry.total_latency_ms ? `${telemetry.total_latency_ms.toFixed(2)} ms` : '233.49 ms'}
                </span>
              </div>
            </div>

            {/* GALLERY */}
            <div className='flex items-center gap-2.5 px-3 py-1.5 rounded-2xl border border-border/60 bg-background/50 shadow-2xs'>
              <div className='flex items-center justify-center size-7 rounded-xl bg-orange-500/10 text-orange-500 dark:bg-amber-500/15 dark:text-amber-400 shrink-0'>
                <Database className='size-3.5' />
              </div>
              <div className='flex flex-col text-left leading-tight'>
                <span className='text-[9px] font-bold tracking-wider text-muted-foreground uppercase font-sans'>GALLERY</span>
                <span className='text-xs font-bold text-foreground font-sans'>
                  {telemetry.gallery_size ? telemetry.gallery_size.toLocaleString() : '1,000'}
                </span>
              </div>
            </div>

            {/* VRAM */}
            <div className='flex items-center gap-2.5 px-3 py-1.5 rounded-2xl border border-border/60 bg-background/50 shadow-2xs'>
              <div className='flex items-center justify-center size-7 rounded-xl bg-orange-500/10 text-orange-500 dark:bg-amber-500/15 dark:text-amber-400 shrink-0'>
                <Cpu className='size-3.5' />
              </div>
              <div className='flex flex-col text-left leading-tight'>
                <span className='text-[9px] font-bold tracking-wider text-muted-foreground uppercase font-sans'>VRAM</span>
                <span className='text-xs font-bold text-foreground font-sans'>
                  {telemetry.vram_allocated_mb ? `${telemetry.vram_allocated_mb.toFixed(1)} MB` : '918.7 MB'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header
