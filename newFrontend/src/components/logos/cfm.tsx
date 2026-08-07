import { SVGProps } from "react";

export default function Cfm(props: SVGProps<SVGSVGElement>) {
  return (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      <path d="M2 12c2-4 4-6 6-6s4 6 6 6 4-6 6-6" />
      <path d="M2 18c2-4 4-6 6-6s4 6 6 6 4-6 6-6" />
      <circle cx="12" cy="12" r="2" fill="currentColor" />
    </svg>
  );
}
