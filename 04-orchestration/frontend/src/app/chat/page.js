import ActiveChat from '@/components/ActiveChat';
import History from '@/components/History';
import styles from './chat.module.css';
import Link from 'next/link';

export default function page() {
  return (
    <div className={styles.chatPage}>
        <nav className={styles.navBar}>
            <Link className={styles.navLink} href={"/"}>RAG Home</Link>
        </nav>
        <div className={styles.chatPageCntr}>
            <History />
            <ActiveChat />
        </div>
    </div>
  )
}
