export const siteConfig = {
  name: "SABER",
  url: "http://localhost:3001",
  getStartedUrl: "/dashboard/format/query",
  ogImage: "/images/og-image.png",
  description:
    "Sensor-Agnostic Bridged Embedding Retrieval — cross-modal satellite image retrieval powered by DOFA ViT, LoRA adapters, and CFM latent bridges.",
  version: "ISRO BAH 2026",
  links: {
    twitter: "https://twitter.com",
    github: "https://github.com/SK8-infi/SABER",
    email: "mailto:team@sentinel8.in",
  },
  pricing: {
    pro: "/dashboard/format/query",
    team: "/dashboard/format/query",
  },
  stats: {
    figma: 14832,
    github: 2078,
    cli: 21675,
    total: "14.8k+",
    updated: "Jul 2026",
    sections: 74,
    illustrations: 23,
    animations: 15,
    templates: 7,
  },
};

export type SiteConfig = typeof siteConfig;
