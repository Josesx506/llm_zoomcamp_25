'use client';

import React from 'react';
import Link from 'next/link';
import styles from '@/styles/historybtn.module.css';
import { usePathname } from "next/navigation";

export default function HistoryBtn({conv_id, title}) {
  const pathname = usePathname();

  return (
    <Link className={`${styles.histLink} ${pathname.endsWith(conv_id) ? styles.active : ''}`} href={`/chat/${conv_id}`}>{title}</Link>
  )
}
