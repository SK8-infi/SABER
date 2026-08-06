// Next Imports
import Link from 'next/link'

// Component Imports
import { Button } from '@/components/ui/button'

const BuyNowButton = () => {
  return (
    <Button
      render={<Link href='https://shadcnstudio.com/#pricing' target='_blank' />}
      className='animate-heartbeat fixed right-15 bottom-8 z-50'
      nativeButton={false}
    >
      Upgrade to Pro
    </Button>
  )
}

export default BuyNowButton
