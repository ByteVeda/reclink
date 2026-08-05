import type { ReactNode } from "react";
import styles from "./Table.module.css";

interface TableProps {
  columns: ReactNode[];
  rows: ReactNode[][];
}

function renderCell(cell: ReactNode): ReactNode {
  if (typeof cell !== "string") return cell;
  const parts = cell.split(/(`[^`]+`)/g);
  if (parts.length === 1) return cell;
  return parts.map((part, i) =>
    part.startsWith("`") && part.endsWith("`") ? (
      <code key={i}>{part.slice(1, -1)}</code>
    ) : (
      part
    ),
  );
}

export default function Table({ columns, rows }: TableProps) {
  return (
    <div className={styles.wrapper}>
      <table className={styles.table}>
        <thead>
          <tr>
            {columns.map((col, i) => (
              <th key={i}>{renderCell(col)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri}>
              {row.map((cell, ci) => (
                <td key={ci}>{renderCell(cell)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
