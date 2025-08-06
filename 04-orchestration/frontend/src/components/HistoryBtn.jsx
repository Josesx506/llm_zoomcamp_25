import React from 'react';
import Link from 'next/link';
import styles from '@/styles/historybtn.module.css';

export default function HistoryBtn({id, title}) {
  return (
    <Link className={styles.histLink} href={`/chat/${id}`}>{title}</Link>
  )
}
