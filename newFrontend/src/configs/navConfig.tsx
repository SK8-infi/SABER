// Third-party Imports
import type * as Icon from 'lucide-react'

type IconName = keyof typeof Icon

export type MenuLeafSubItem = {
  label: string
  href: string
  activePath?: string
  badge?: string
  badgeClassName?: string
  target?: '_blank' | '_self' | '_parent' | '_top'
}

export type MenuGroupSubItem = {
  label: string
  childItems: MenuLeafSubItem[]
}

export type MenuSubItem = MenuLeafSubItem | MenuGroupSubItem

export type MenuItem = {
  icon: IconName
  label: string
} & (
  | {
      href: string
      badge?: string
      badgeClassName?: string
      childItems?: never
      target?: '_blank' | '_self' | '_parent' | '_top'
    }
  | {
      href?: never
      badge?: string
      badgeClassName?: string
      childItems: MenuSubItem[]
    }
)

export type NavItem = {
  groupLabel?: string
  items: MenuItem[]
}

export const navItems: NavItem[] = [
  {
    groupLabel: 'Format',
    items: [
      {
        icon: 'Share2',
        label: 'Interactive Query Space',
        href: '/dashboard/format/embeddings'
      },
      {
        icon: 'FileText',
        label: 'Classic Query Inspector',
        href: '/dashboard/format/query'
      },
      {
        icon: 'Database',
        label: 'DSRSID 1,000 Scene Search',
        href: '/dashboard/format/dsrsid-search'
      },
      {
        icon: 'Sliders',
        label: 'Ablation Studies',
        href: '/dashboard/format/abliation'
      },
      {
        icon: 'Activity',
        label: 'Training Telemetry',
        href: '/dashboard/format/training'
      },
      {
        icon: 'CloudOff',
        label: 'Cloud-Free Demonstration',
        href: '/dashboard/format/cloud-free'
      }
    ]
  }
]
