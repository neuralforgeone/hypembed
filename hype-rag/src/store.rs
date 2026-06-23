use std::path::Path;

use rusqlite::{params, Connection};

use crate::error::{HypeRagError, Result};

const CONFIG_KEY: &str = "model_dir";

/// SQLite-backed chunk store with brute-force vector retrieval.
pub struct ChunkStore {
    conn: Connection,
}

impl ChunkStore {
    pub fn open(data_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(data_dir)?;
        let db_path = data_dir.join("index.sqlite");
        let conn = Connection::open(db_path)?;
        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL
            );",
        )?;
        Ok(())
    }

    pub fn set_model_dir(&self, model_dir: &Path) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![CONFIG_KEY, model_dir.display().to_string()],
        )?;
        Ok(())
    }

    pub fn model_dir(&self) -> Result<Option<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT value FROM meta WHERE key = ?1")?;
        let mut rows = stmt.query(params![CONFIG_KEY])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row.get(0)?))
        } else {
            Ok(None)
        }
    }

    pub fn clear_chunks(&self) -> Result<()> {
        self.conn.execute("DELETE FROM chunks", [])?;
        Ok(())
    }

    pub fn insert_chunk(
        &self,
        path: &str,
        chunk_index: usize,
        text: &str,
        embedding: &[f32],
    ) -> Result<()> {
        let bytes: Vec<u8> = embedding.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.conn.execute(
            "INSERT INTO chunks(path, chunk_index, text, embedding) VALUES (?1, ?2, ?3, ?4)",
            params![path, chunk_index as i64, text, bytes],
        )?;
        Ok(())
    }

    pub fn all_chunks(&self) -> Result<Vec<StoredChunk>> {
        let mut stmt = self.conn.prepare(
            "SELECT path, chunk_index, text, embedding FROM chunks ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            let path: String = row.get(0)?;
            let chunk_index: i64 = row.get(1)?;
            let text: String = row.get(2)?;
            let blob: Vec<u8> = row.get(3)?;
            Ok((path, chunk_index, text, blob))
        })?;

        let mut out = Vec::new();
        for row in rows {
            let (path, chunk_index, text, blob) = row?;
            if blob.len() % 4 != 0 {
                return Err(HypeRagError::Other(
                    "invalid embedding blob length".into(),
                ));
            }
            let embedding: Vec<f32> = blob
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            out.push(StoredChunk {
                path,
                chunk_index: chunk_index as usize,
                text,
                embedding,
            });
        }
        Ok(out)
    }

    pub fn chunk_count(&self) -> Result<usize> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;
        Ok(count as usize)
    }
}

#[derive(Debug, Clone)]
pub struct StoredChunk {
    pub path: String,
    pub chunk_index: usize,
    pub text: String,
    pub embedding: Vec<f32>,
}