// Third-party Imports
import { CheckIcon, EyeIcon, XIcon } from 'lucide-react'

// Type Imports
import type { AppUser } from '@/types/apps/user-types'

// Component Imports
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'

// Config Imports
import { getInitialsFromName } from '@/configs/mailConfig'

// Util Imports
import { cn } from '@/lib/utils'

export interface SceneTag {
  label: string
  checked?: boolean
}

export interface SceneData {
  sceneId: string
  progressColor: string
  progressWidth: string
  jaccard: string
  tags: SceneTag[]
  rank?: string
  matched?: boolean
  matchScore?: string
}

const defaultSceneData: SceneData = {
  sceneId: 'S2A_MSIL2A_20170803T09403...',
  progressColor: 'bg-[#FBBA72]',
  progressWidth: 'w-[30%]',
  jaccard: '25%',
  tags: [
    { label: 'Urban fabric' },
    { label: 'Arable land' },
    { label: 'Broad-leaved forest', checked: true }
  ]
}

export interface UserViewLeftPanelProps {
  user: AppUser
  sceneData?: SceneData
  onEdit?: () => void
  onInspect?: () => void
  className?: string
}

export function UserViewLeftPanel({ user, sceneData = defaultSceneData, onInspect, className }: UserViewLeftPanelProps) {
  return (
    <Card className={cn('h-full flex flex-col overflow-hidden shadow-xs border-border/80', className)}>
      <CardContent className='flex flex-col flex-1 justify-between p-3.5 space-y-2.5'>
        {/* Profile / Scene Image Container */}
        <div className='relative w-full aspect-square rounded-lg overflow-hidden border border-border/60 bg-muted/20 group shrink-0'>
          {user.avatar ? (
            <img
              src={user.avatar}
              alt={user.name}
              className='w-full h-full object-cover transition-transform duration-300 group-hover:scale-105'
            />
          ) : (
            <div className='w-full h-full bg-gradient-to-br from-neutral-800 via-neutral-900 to-black flex items-center justify-center text-2xl font-semibold text-foreground'>
              {getInitialsFromName(user.name)}
            </div>
          )}

          {/* Rank badge — top left */}
          {sceneData.rank && (
            <span className='absolute top-1.5 left-1.5 bg-black/85 text-[#FBBA72] text-[10px] font-bold px-1.5 py-0.5 rounded font-sans border border-[#FBBA72]/30'>
              {sceneData.rank}
            </span>
          )}

          {/* Match score badge — top right */}
          {sceneData.matchScore && (
            <span className='absolute top-1.5 right-1.5 bg-rose-500/90 text-white text-[10px] font-bold px-1.5 py-0.5 rounded font-sans'>
              {sceneData.matchScore}
            </span>
          )}

          {/* Correct / Wrong symbol — bottom left */}
          {sceneData.matched !== undefined && (
            <div className='absolute bottom-1.5 left-1.5'>
              {sceneData.matched ? (
                <div className='size-5 rounded-full bg-[#FBBA72] flex items-center justify-center shadow-sm'>
                  <CheckIcon className='size-3 text-black stroke-[3]' />
                </div>
              ) : (
                <div className='size-5 rounded-full bg-rose-500 flex items-center justify-center shadow-sm'>
                  <XIcon className='size-3 text-white stroke-[3]' />
                </div>
              )}
            </div>
          )}
        </div>

        {/* Structured metadata area with 100% consistent alignment across cards */}
        <div className='w-full flex flex-col flex-1 justify-between gap-2 text-left pt-1'>
          {/* Scene ID Header & #FBBA72 Progress Line */}
          <div className='w-full flex flex-col gap-1 shrink-0'>
            <span className='text-xs font-semibold text-foreground truncate w-full tracking-tight leading-none' title={sceneData.sceneId}>
              {sceneData.sceneId}
            </span>
            <div className='w-full h-1 bg-muted/80 rounded-full overflow-hidden'>
              <div
                className={cn('h-full rounded-full transition-all duration-300', sceneData.progressWidth)}
                style={{ backgroundColor: '#FBBA72' }}
              />
            </div>
          </div>

          {/* Jaccard Score Row using #FBBA72 */}
          <div className='w-full flex items-center justify-between text-xs shrink-0 py-0.5'>
            <div className='flex items-center gap-1.5'>
              <span className='text-muted-foreground text-xs font-medium'>Jaccard</span>
              {sceneData.matched ? (
                <Badge
                  variant='outline'
                  className='bg-[#FBBA72]/10 text-[#FBBA72] border-[#FBBA72]/30 text-[10px] font-semibold px-1.5 py-0 rounded flex items-center gap-1'
                >
                  <CheckIcon className='size-3 text-[#FBBA72]' />
                  MATCH
                </Badge>
              ) : (
                <Badge
                  variant='outline'
                  className='bg-rose-500/10 text-rose-500 border-rose-500/30 text-[10px] font-semibold px-1.5 py-0 rounded flex items-center gap-1'
                >
                  <XIcon className='size-3 text-rose-500' />
                  NO MATCH
                </Badge>
              )}
            </div>
            <span className={cn('font-bold text-xs', sceneData.matched ? 'text-[#FBBA72]' : 'text-rose-500/90')}>{sceneData.jaccard}</span>
          </div>

          {/* Land classification tags (Fixed 72px height block across all cards) */}
          <div className='w-full h-[72px] flex flex-wrap content-start items-start gap-1 overflow-y-auto no-scrollbar shrink-0 py-0.5'>
            {sceneData.tags.map((tag, idx) =>
              tag.checked ? (
                <Badge
                  key={idx}
                  variant='outline'
                  className='bg-[#FBBA72]/10 text-[#FBBA72] border-[#FBBA72]/30 text-[10px] font-medium px-2 py-0.5 rounded-md flex items-center gap-1 shrink-0'
                >
                  <CheckIcon className='size-3 text-[#FBBA72]' />
                  {tag.label}
                </Badge>
              ) : (
                <Badge
                  key={idx}
                  variant='secondary'
                  className='bg-muted/70 text-muted-foreground hover:bg-muted border-transparent text-[10px] font-medium px-2 py-0.5 rounded-md shrink-0'
                >
                  {tag.label}
                </Badge>
              )
            )}
          </div>

          {/* Inspect button */}
          <Button
            size='sm'
            onClick={onInspect}
            className='w-full flex items-center justify-center gap-1.5 h-8 text-xs font-medium mt-auto shrink-0 shadow-xs cursor-pointer'
          >
            <EyeIcon className='size-3.5' />
            Inspect
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
