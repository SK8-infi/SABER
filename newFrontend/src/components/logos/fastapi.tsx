import React from "react";

export default function FastAPI(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="currentColor"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm.83 4.316l2.19 5.86h-2.99l.8 4.54-5.01-6.19h3.04l-1.03-4.21h3z" />
    </svg>
  );
}
