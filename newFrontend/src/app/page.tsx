import Architecture from "@/components/sections/architecture/default"
import Hero from "@/components/sections/hero/default"
import Items from "@/components/sections/items/default"
import Logos from "@/components/sections/logos/default"
import Navbar from "@/components/sections/navbar/default"
import Stats from "@/components/sections/stats/default"
import { LayoutLines } from "@/components/ui/layout-lines"

export default function Home() {
  return (
    <main className="bg-background text-foreground min-h-screen w-full">
      <LayoutLines />
      <Navbar />
      <Hero />
      <Logos />
      <Architecture />
      <Stats />
      <Items />
    </main>
  )
}
