import styles from "./page.module.css";
import Link from "next/link";

export default function Home() {
  return (
    <div className={styles.page}>
      <main className={styles.main}>
        <h3>Welcome to the RAG chat</h3>
        <div>Each conversation has a rate limit of 10 requests per minute.</div>
        <div className={styles.chatBtn}><Link href={"/chat"}>Start Chat</Link></div>
      </main>
    </div>
  );
}
