'use client';

import styles from '@/styles/history.module.css';
import Link from 'next/link';
import { PiChatTeardropTextBold } from "react-icons/pi";
import { useState } from 'react';
import HistoryBtn from './HistoryBtn';

export default function History() {
  const [convos,setConvos] = useState([
    {"id":"1","title":"This is chat 1, It's a long name for a chat"},
    {"id":"2","title":"This is chat 2"},
    {"id":"3","title":"This is chat 3"},
    {"id":"4","title":"This is chat 4"},
    {"id":"5","title":"This is chat 5"},
    {"id":"6","title":"This is chat 6"},
    {"id":"7","title":"This is chat 7"},
    {"id":"8","title":"This is chat 8"},
    {"id":"9","title":"This is chat 9"}]);

  return (
    <aside className={styles.chatHistory}>
      <div className={styles.newChat}>
        <Link className={styles.newChatBtn} href={"#"}>
          <PiChatTeardropTextBold size={"1rem"} /> <span>New Chat</span>
        </Link>
      </div>
      <div className={styles.prevChats}>
        <h3>Chats</h3>
        <div className={styles.prevChatList}>
          {convos.map((conv)=>(<HistoryBtn key={conv.id} {...conv} />))}
        </div>
      </div>
    </aside>
  )
}
