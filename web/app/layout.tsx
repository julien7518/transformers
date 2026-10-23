import type {Metadata} from 'next';
import './globals.css';
export const metadata: Metadata = {title: 'HEIN ? — Le SAV des expressions', description: 'Du français standard au langage Gen Z et retour. Un traducteur étudiant fait maison, parfois un peu à côté.'};
export default function RootLayout({children}: {children: React.ReactNode}) {return <html lang="fr"><body>{children}</body></html>;}
