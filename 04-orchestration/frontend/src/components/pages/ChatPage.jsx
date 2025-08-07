import ActiveChat from '@/components/ActiveChat';
import History from '@/components/History';
import styles from '@/styles/pages/chat.module.css';
import Link from 'next/link';

export default function ChatPage({ conv_id }) {
  return (
    <div className={styles.chatPage}>
        <nav className={styles.navBar}>
            <Link className={styles.navLink} href={"/"}>RAG Home</Link>
        </nav>
        <div className={styles.chatPageCntr}>
            <History />
            <ActiveChat conv_id={conv_id} />
        </div>
    </div>
  )
}
