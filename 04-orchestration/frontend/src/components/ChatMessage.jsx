import styles from '@/styles/chatmsg.module.css';
import Link from 'next/link';
import { AiOutlineDislike, AiOutlineLike } from "react-icons/ai";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function ChatMessage({ msg_id, query, response }) {
  return (
    <div className={styles.msgCntr}>
      <div className={styles.query}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{query}</ReactMarkdown>
      </div>
      <div className={styles.response}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{response}</ReactMarkdown>
      </div>
      <div className={styles.review}>
        <Link href={`#/${msg_id}`}><AiOutlineLike /></Link>
        <Link href={`#/${msg_id}`}><AiOutlineDislike /></Link>
      </div>
    </div>
  )
}
