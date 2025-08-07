import React from 'react';
import Link from 'next/link';
import styles from '@/styles/historybtn.module.css';

export default function HistoryBtn({conv_id, title}) {
  return (
    <Link className={styles.histLink} href={`/chat/${conv_id}`}>{title}</Link>
  )
}
