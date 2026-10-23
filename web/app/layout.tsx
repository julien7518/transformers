import type {Metadata} from 'next';
import './globals.css';
export const metadata: Metadata = {title: 'Entre nous — Le français change de ton', description: 'Un laboratoire de traduction français standard et Gen Z, avec un Transformer entraîné depuis zéro.'};
export default function RootLayout({children}: {children: React.ReactNode}) {return <html lang="fr"><body>{children}</body></html>;}
