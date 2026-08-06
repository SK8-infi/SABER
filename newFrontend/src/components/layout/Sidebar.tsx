'use client'

// React Imports
import { type ComponentType } from 'react'

import { useState } from 'react'

// Next Imports
import Link from 'next/link'
import { usePathname, useSearchParams } from 'next/navigation'

// Third-party Imports
import * as Icon from 'lucide-react'
import { ChevronRightIcon, SquareArrowOutUpRightIcon, RotateCw } from 'lucide-react'

// Type Imports
import type { MenuGroupSubItem, MenuItem, MenuSubItem } from '@/configs/navConfig'

// Component Imports
import LogoSvg from '@/assets/svg/logo'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { Slider } from '@/components/ui/slider'
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuBadge,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem
} from '@/components/ui/sidebar'

// Config Imports
import { navItems } from '@/configs/navConfig'
import themeConfig from '@/configs/themeConfig'

// Context Imports
import { useRetrievalParams } from '@/contexts/retrieval-params-context'

// Util Imports
import { cn } from '@/lib/utils'

const isSubGroup = (item: MenuSubItem): item is MenuGroupSubItem => 'childItems' in item

const isExternalLink = (href: string) => href.startsWith('http://') || href.startsWith('https://')

function isLinkActive(
  href: string,
  activePath: string | undefined,
  pathname: string,
  searchParams: Pick<URLSearchParams, 'get'>
): boolean {
  if (activePath) {
    return pathname.startsWith(activePath)
  }

  if (href.includes('?')) {
    const [hrefPath, hrefQuery] = href.split('?')

    if (pathname !== hrefPath) return false

    const hrefParams = new URLSearchParams(hrefQuery)

    for (const [key, value] of hrefParams.entries()) {
      if (searchParams.get(key) !== value) return false
    }

    return true
  }

  return pathname === href
}

const SidebarGroupedMenuItems = ({
  data,
  groupLabel,
  pathname,
  searchParams
}: {
  data: MenuItem[]
  groupLabel?: string
  pathname: string
  searchParams: Pick<URLSearchParams, 'get'>
}) => {
  return (
    <SidebarGroup>
      {groupLabel && (
        <SidebarGroupLabel className='text-sidebar-foreground/50 tracking-wider uppercase'>
          {groupLabel}
        </SidebarGroupLabel>
      )}
      <SidebarGroupContent>
        <SidebarMenu>
          {data.map(item => {
            const Tag = item.icon ? (Icon[item.icon] as ComponentType) : null

            const isChildActive =
              item.childItems?.some(subItem =>
                isSubGroup(subItem)
                  ? subItem.childItems.some(leaf => isLinkActive(leaf.href, leaf.activePath, pathname, searchParams))
                  : isLinkActive(subItem.href, subItem.activePath, pathname, searchParams)
              ) ?? false

            return item.childItems ? (
              <Collapsible className='group/collapsible' key={item.label}>
                <SidebarMenuItem>
                  <CollapsibleTrigger
                    render={
                      <SidebarMenuButton
                        tooltip={item.label}
                        isActive={isChildActive}
                        className='data-active:bg-primary/5!'
                      />
                    }
                  >
                    {Tag && <Tag />}
                    <span className={cn('min-w-0 flex-1 truncate', item.badge && 'pr-14')}>{item.label}</span>
                    {item.badge && (
                      <SidebarMenuBadge
                        className={cn(
                          'bg-primary/10 max-w-24 truncate rounded-full px-1.5 font-normal',
                          item.badgeClassName
                        )}
                      >
                        {item.badge}
                      </SidebarMenuBadge>
                    )}
                    <ChevronRightIcon className='ml-auto transition-transform duration-200 group-data-open/collapsible:rotate-90' />
                  </CollapsibleTrigger>
                  <CollapsibleContent className='h-(--collapsible-panel-height) overflow-hidden transition-all duration-200 data-ending-style:h-0 data-starting-style:h-0'>
                    <SidebarMenuSub>
                      {item.childItems.map(subItem =>
                        isSubGroup(subItem) ? (
                          <Collapsible className='group/subcollapsible' key={subItem.label}>
                            <SidebarMenuSubItem>
                              <CollapsibleTrigger
                                nativeButton={false}
                                render={
                                  <SidebarMenuSubButton
                                    className='data-active:bg-primary/10! justify-between'
                                    isActive={subItem.childItems.some(leaf =>
                                      isLinkActive(leaf.href, leaf.activePath, pathname, searchParams)
                                    )}
                                  />
                                }
                              >
                                {subItem.label}
                                <ChevronRightIcon className='ml-auto shrink-0 transition-transform duration-200 group-data-open/subcollapsible:rotate-90' />
                              </CollapsibleTrigger>
                              <CollapsibleContent className='h-(--collapsible-panel-height) overflow-hidden transition-all duration-200 data-ending-style:h-0 data-starting-style:h-0'>
                                <SidebarMenuSub className='mx-0'>
                                  {subItem.childItems.map(leaf => (
                                    <SidebarMenuSubItem key={leaf.label}>
                                      <SidebarMenuSubButton
                                        className='data-active:bg-primary/10! justify-between'
                                        render={<Link href={leaf.href} target={leaf.target} />}
                                        isActive={isLinkActive(leaf.href, leaf.activePath, pathname, searchParams)}
                                      >
                                        <span
                                          className={cn(
                                            'min-w-0 flex-1 truncate',
                                            leaf.badge && isExternalLink(leaf.href) && 'pr-8',
                                            leaf.badge && !isExternalLink(leaf.href) && 'pr-14',
                                            !leaf.badge && isExternalLink(leaf.href) && 'pr-6'
                                          )}
                                        >
                                          {leaf.label}
                                        </span>
                                        {leaf.badge && (
                                          <SidebarMenuBadge
                                            className={cn(
                                              'bg-primary/10 max-w-24 truncate rounded-full px-1.5 font-normal',
                                              isExternalLink(leaf.href) && 'right-6',
                                              leaf.badgeClassName
                                            )}
                                          >
                                            {leaf.badge}
                                          </SidebarMenuBadge>
                                        )}
                                        {isExternalLink(leaf.href) && (
                                          <SquareArrowOutUpRightIcon className='ml-auto size-3.5! shrink-0 opacity-50' />
                                        )}
                                      </SidebarMenuSubButton>
                                    </SidebarMenuSubItem>
                                  ))}
                                </SidebarMenuSub>
                              </CollapsibleContent>
                            </SidebarMenuSubItem>
                          </Collapsible>
                        ) : (
                          <SidebarMenuSubItem key={subItem.label}>
                            <SidebarMenuSubButton
                              className='data-active:bg-primary/10! justify-between'
                              render={<Link href={subItem.href} target={subItem.target} />}
                              isActive={isLinkActive(subItem.href, subItem.activePath, pathname, searchParams)}
                            >
                              <span
                                className={cn(
                                  'min-w-0 flex-1 truncate',
                                  subItem.badge && isExternalLink(subItem.href) && 'pr-8',
                                  subItem.badge && !isExternalLink(subItem.href) && 'pr-14',
                                  !subItem.badge && isExternalLink(subItem.href) && 'pr-6'
                                )}
                              >
                                {subItem.label}
                              </span>
                              {subItem.badge && (
                                <SidebarMenuBadge
                                  className={cn(
                                    'bg-primary/10 max-w-24 truncate rounded-full px-1.5 font-normal',
                                    isExternalLink(subItem.href) && 'right-6',
                                    subItem.badgeClassName
                                  )}
                                >
                                  {subItem.badge}
                                </SidebarMenuBadge>
                              )}
                              {isExternalLink(subItem.href) && (
                                <SquareArrowOutUpRightIcon className='ml-auto size-3.5! shrink-0 opacity-50' />
                              )}
                            </SidebarMenuSubButton>
                          </SidebarMenuSubItem>
                        )
                      )}
                    </SidebarMenuSub>
                  </CollapsibleContent>
                </SidebarMenuItem>
              </Collapsible>
            ) : (
              <SidebarMenuItem key={item.label}>
                <SidebarMenuButton
                  tooltip={item.label}
                  render={<Link href={item.href} target={item.target} />}
                  isActive={pathname === item.href}
                  className='data-active:bg-primary/10!'
                >
                  {Tag && <Tag />}
                  <span
                    className={cn(
                      'min-w-0 flex-1 truncate',
                      item.badge && isExternalLink(item.href) && 'pr-8',
                      item.badge && !isExternalLink(item.href) && 'pr-14',
                      !item.badge && isExternalLink(item.href) && 'pr-6'
                    )}
                  >
                    {item.label}
                  </span>
                  {item.badge && (
                    <SidebarMenuBadge
                      className={cn(
                        'bg-primary/10 max-w-24 truncate rounded-full px-1.5 font-normal',
                        isExternalLink(item.href) && 'right-6',
                        item.badgeClassName
                      )}
                    >
                      {item.badge}
                    </SidebarMenuBadge>
                  )}
                  {isExternalLink(item.href) && (
                    <SquareArrowOutUpRightIcon className='ml-auto size-3.5! shrink-0 opacity-50' />
                  )}
                </SidebarMenuButton>
              </SidebarMenuItem>
            )
          })}
        </SidebarMenu>
      </SidebarGroupContent>
    </SidebarGroup>
  )
}

const SidebarLayout = () => {
  const pathname = usePathname()
  const searchParams = useSearchParams()
  const { params, setParams } = useRetrievalParams()

  // ── Local UI display labels (mapped from context params) ────
  const datasetLabel = params.dataset === 'ben14k' ? 'BEN-14K — Sentinel-1/2' : 'DSRSID — Gaofen-1'

  const sourceLabel = params.dataset === 'ben14k'
    ? (params.srcMod === 's1' ? 'Sentinel-1 SAR (2ch)' : 'Sentinel-2 MS (12ch)')
    : (params.srcMod === 'pan' ? 'Gaofen-1 PAN (1ch)' : 'Gaofen-1 MS (4ch)')

  const targetLabel = params.dataset === 'ben14k'
    ? (params.tgtMod === 's2' ? 'Sentinel-2 MS (Cross)' : 'Sentinel-1 SAR (Same)')
    : (params.tgtMod === 'ms' ? 'Gaofen-1 MS (Cross)' : 'Gaofen-1 PAN (Same)')

  const handleDatasetChange = (label: string) => {
    if (label === 'BEN-14K — Sentinel-1/2') {
      setParams({ dataset: 'ben14k', srcMod: 's1', tgtMod: 's2' })
    } else {
      setParams({ dataset: 'dsrsid', srcMod: 'pan', tgtMod: 'ms' })
    }
  }

  const handleSourceChange = (label: string) => {
    if (params.dataset === 'ben14k') {
      setParams({ srcMod: label.includes('SAR') ? 's1' : 's2' })
    } else {
      setParams({ srcMod: label.includes('PAN') ? 'pan' : 'ms' })
    }
  }

  const handleTargetChange = (label: string) => {
    if (params.dataset === 'ben14k') {
      setParams({ tgtMod: label.includes('MS') ? 's2' : 's1' })
    } else {
      setParams({ tgtMod: label.includes('MS') ? 'ms' : 'pan' })
    }
  }

  const [advancedOpen, setAdvancedOpen] = useState(true)
  const isAblation = pathname.includes('/abliation')

  // Nav groups rendered in the sidebar.
  const formatGroups = navItems.filter(item => item.groupLabel === 'Format')

  return (
    <Sidebar collapsible='icon' variant='sidebar'>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              size='lg'
              className='gap-2.5 bg-transparent! [&>svg]:size-8'
              render={<Link href={`${themeConfig.homePageUrl}`} />}
            >
              <LogoSvg className='[&_rect]:fill-sidebar [&_rect:first-child]:fill-primary' />
              <div className='flex flex-col items-start'>
                <span className='text-lg font-bold tracking-tight text-nowrap'>{themeConfig.templateName}</span>
                <span className='text-xs font-light text-nowrap text-sidebar-foreground/60'>Scientific Retrieval</span>
              </div>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent className='group-data-[collapsible=icon]:overflow-y-auto'>
        {/* FORMAT Section at the top */}
        {formatGroups.map((navItem, index) => {
          return (
            <SidebarGroupedMenuItems
              key={navItem.groupLabel || index}
              data={navItem.items}
              groupLabel={navItem.groupLabel}
              pathname={pathname}
              searchParams={searchParams}
            />
          )
        })}

        {/* Dataset, Source, Target Gallery, Scene Index Controls below Format */}
        <SidebarGroup className='gap-3 px-3 py-2 group-data-[collapsible=icon]:hidden'>
          {/* DATASET */}
          <div className='flex flex-col gap-1.5'>
            <SidebarGroupLabel className='text-sidebar-foreground/50 h-auto p-0 text-xs font-semibold tracking-wider uppercase'>
              DATASET
            </SidebarGroupLabel>
            <Select value={datasetLabel} onValueChange={val => val && handleDatasetChange(val)}>
              <SelectTrigger className='border-border/60 bg-background/50 h-9 w-full rounded-lg'>
                <SelectValue placeholder='Select Dataset' />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  <SelectItem value='BEN-14K — Sentinel-1/2'>BEN-14K — Sentinel-1/2</SelectItem>
                  <SelectItem value='DSRSID — Gaofen-1'>DSRSID — Gaofen-1</SelectItem>
                </SelectGroup>
              </SelectContent>
            </Select>
          </div>

          <Separator className='my-1 bg-border/40' />

          {/* SOURCE — hidden on ablation route */}
          {!isAblation && (
          <div className='flex flex-col gap-1.5'>
            <SidebarGroupLabel className='text-sidebar-foreground/50 h-auto p-0 text-xs font-semibold tracking-wider uppercase'>
              SOURCE
            </SidebarGroupLabel>
            <Select value={sourceLabel} onValueChange={val => val && handleSourceChange(val)}>
              <SelectTrigger className='border-border/60 bg-background/50 h-9 w-full rounded-lg'>
                <SelectValue placeholder='Select Source' />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  {params.dataset === 'ben14k' ? (
                    <>
                      <SelectItem value='Sentinel-1 SAR (2ch)'>Sentinel-1 SAR (2ch)</SelectItem>
                      <SelectItem value='Sentinel-2 MS (12ch)'>Sentinel-2 MS (12ch)</SelectItem>
                    </>
                  ) : (
                    <>
                      <SelectItem value='Gaofen-1 PAN (1ch)'>Gaofen-1 PAN (1ch)</SelectItem>
                      <SelectItem value='Gaofen-1 MS (4ch)'>Gaofen-1 MS (4ch)</SelectItem>
                    </>
                  )}
                </SelectGroup>
              </SelectContent>
            </Select>
          </div>
          )}

          {/* TARGET GALLERY — hidden (auto-derived for both same+cross modal) */}
          {false && (
          <div className='flex flex-col gap-1.5 mt-1'>
            <SidebarGroupLabel className='text-sidebar-foreground/50 h-auto p-0 text-xs font-semibold tracking-wider uppercase'>
              TARGET GALLERY
            </SidebarGroupLabel>
            <Select value={targetLabel} onValueChange={val => val && handleTargetChange(val)}>
              <SelectTrigger className='border-border/60 bg-background/50 h-9 w-full rounded-lg'>
                <SelectValue placeholder='Select Target Gallery' />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  {params.dataset === 'ben14k' ? (
                    <>
                      <SelectItem value='Sentinel-2 MS (Cross)'>Sentinel-2 MS (Cross)</SelectItem>
                      <SelectItem value='Sentinel-1 SAR (Same)'>Sentinel-1 SAR (Same)</SelectItem>
                    </>
                  ) : (
                    <>
                      <SelectItem value='Gaofen-1 MS (Cross)'>Gaofen-1 MS (Cross)</SelectItem>
                      <SelectItem value='Gaofen-1 PAN (Same)'>Gaofen-1 PAN (Same)</SelectItem>
                    </>
                  )}
                </SelectGroup>
              </SelectContent>
            </Select>
          </div>
          )}

          <Separator className='my-1 bg-border/40' />

          {/* SCENE INDEX */}
          <div className='flex flex-col gap-2'>
            <div className='text-sidebar-foreground/50 flex items-center text-xs font-semibold tracking-wider uppercase'>
              <span>SCENE INDEX</span>
              <span className='ml-1.5 font-bold text-amber-500'>#{params.qIdx}</span>
            </div>
            <Input
              type='number'
              value={params.qIdx}
              onChange={e => setParams({ qIdx: parseInt(e.target.value) || 0 })}
              className='border-border/60 bg-background/50 h-9 w-full rounded-lg px-3 text-sm'
            />
            <Button
              variant='outline'
              size='sm'
              onClick={() => setParams({ qIdx: Math.floor(Math.random() * 2965) })}
              className='border-border/60 hover:bg-accent h-9 w-full justify-center gap-2 rounded-lg text-sm font-medium'
            >
              <RotateCw className='size-4' />
              <span>Random Scene</span>
            </Button>

            {/* TOP-K SLIDER — hidden on ablation route */}
            {!isAblation && (
            <div className='flex flex-col gap-2 pt-2'>
              <div className='text-sidebar-foreground/50 flex items-center justify-between text-xs font-semibold tracking-wider uppercase'>
                <span>TOP-K</span>
                <span className='text-primary font-bold'>{params.topK}</span>
              </div>
              <Slider
                value={[params.topK]}
                onValueChange={(val: any) => {
                  const v = Array.isArray(val) ? val[0] : val
                  setParams({ topK: v })
                }}
                min={1}
                max={20}
                step={1}
                className='py-1'
              />
            </div>
            )}
          </div>

          {/* ADVANCED SECTION — hidden on ablation route */}
          {!isAblation && (
          <>
          <Separator className='my-1 bg-border/40' />

          <div className='flex flex-col gap-3'>
            {/* Header */}
            <button
              onClick={() => setAdvancedOpen(o => !o)}
              className='flex items-center justify-between w-full text-sidebar-foreground/70 hover:text-sidebar-foreground transition-colors'
            >
              <span className='text-sm font-medium'>Advanced</span>
              <ChevronRightIcon
                className={`size-4 transition-transform duration-200 ${advancedOpen ? '-rotate-90' : 'rotate-90'}`}
              />
            </button>

            {advancedOpen && (
              <div className='flex flex-col gap-4'>
                {/* CFM Bridge toggle */}
                <div className='flex items-center justify-between'>
                  <span className='text-sm font-medium text-sidebar-foreground'>CFM Bridge</span>
                  <button
                    onClick={() => setParams({ bridge: !params.bridge })}
                    className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 focus:outline-none ${
                      params.bridge ? 'bg-[#FBBA72]/20' : 'bg-muted'
                    }`}
                    role='switch'
                    aria-checked={params.bridge}
                  >
                    <span
                      className={`pointer-events-none inline-block size-5 rounded-full shadow-sm ring-0 transition-transform duration-200 ${
                        params.bridge ? 'translate-x-5 bg-[#FBBA72]' : 'translate-x-0 bg-muted-foreground/50'
                      }`}
                    />
                  </button>
                </div>

                {/* ODE STEPS slider */}
                <div className='flex flex-col gap-2'>
                  <div className='flex items-center gap-2 text-xs font-semibold tracking-wider uppercase text-sidebar-foreground/50'>
                    <span>ODE STEPS:</span>
                    <span className='text-white font-bold'>{params.odeSteps}</span>
                    <span className='text-sidebar-foreground/40 normal-case font-normal tracking-normal'>
                      ~{(params.odeSteps * 5.01).toFixed(1)} MS EST.
                    </span>
                  </div>
                  <Slider
                    value={[params.odeSteps]}
                    onValueChange={(val: any) => setParams({ odeSteps: Array.isArray(val) ? val[0] : val })}
                    min={1}
                    max={15}
                    step={1}
                    className='py-1 [&_[data-slot=slider-track]]:bg-muted/60 [&_[data-slot=slider-range]]:bg-white [&_[data-slot=slider-thumb]]:bg-white [&_[data-slot=slider-thumb]]:border-white [&_[data-slot=slider-thumb]]:shadow-white/40 [&_[data-slot=slider-thumb]]:size-4'
                  />
                  <div className='flex items-center justify-between text-[10px] text-sidebar-foreground/40 font-medium'>
                    <span>Fast (1)</span>
                    <span>Balanced (5)</span>
                    <span>Precise (15)</span>
                  </div>
                </div>

                {/* Jaccard Reranking toggle */}
                <div className='flex items-center justify-between'>
                  <span className='text-sm font-medium text-sidebar-foreground'>Jaccard Reranking</span>
                  <button
                    onClick={() => setParams({ rerank: !params.rerank })}
                    className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 focus:outline-none ${
                      params.rerank ? 'bg-[#FBBA72]/20' : 'bg-muted'
                    }`}
                    role='switch'
                    aria-checked={params.rerank}
                  >
                    <span
                      className={`pointer-events-none inline-block size-5 rounded-full shadow-sm ring-0 transition-transform duration-200 ${
                        params.rerank ? 'translate-x-5 bg-[#FBBA72]' : 'translate-x-0 bg-muted-foreground/50'
                      }`}
                    />
                  </button>
                </div>
              </div>
            )}
          </div>

          <Separator className='my-1 bg-border/40' />
          </>
          )}
        </SidebarGroup>

        {/* Remaining Navigation Sections — removed */}
      </SidebarContent>
    </Sidebar>
  )
}

export default SidebarLayout
