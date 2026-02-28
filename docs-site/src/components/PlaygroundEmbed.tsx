import type {ReactNode} from 'react';

interface PlaygroundEmbedProps {
  page?: string;
  height?: string;
}

const STREAMLIT_BASE_URL =
  process.env.NODE_ENV === 'development'
    ? 'http://localhost:8501'
    : 'https://reclink-playground.streamlit.app';

export default function PlaygroundEmbed({
  page,
  height = '750px',
}: PlaygroundEmbedProps): ReactNode {
  const url = page ? `${STREAMLIT_BASE_URL}/${page}` : STREAMLIT_BASE_URL;

  return (
    <div className="playground-container">
      <iframe
        src={url}
        style={{width: '100%', height, border: 'none'}}
        title="reclink Interactive Playground"
        allow="clipboard-read; clipboard-write"
        loading="lazy"
      />
    </div>
  );
}
