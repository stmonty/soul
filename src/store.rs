//! Single-writer durable snapshot store for the prototype controller.
//! Persist before publishing state to readers. No external commands are executed.
use crate::model::Cluster;
use rusqlite::{Connection, OptionalExtension};
use std::path::Path;

pub struct Store {
    connection: Connection,
    state: Cluster,
}

impl Store {
    pub fn open(path: impl AsRef<Path>, initial: Cluster) -> Result<Self, String> {
        initial.validate()?;
        let expected_mode = initial.mode;
        let connection = Connection::open(path).map_err(|e| e.to_string())?;
        connection.execute_batch("PRAGMA locking_mode=EXCLUSIVE; PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS snapshot (id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL, body TEXT NOT NULL);")
            .map_err(|e| e.to_string())?;
        let saved: Option<(u32, String)> = connection
            .query_row("SELECT version, body FROM snapshot WHERE id=1", [], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .optional()
            .map_err(|e| e.to_string())?;
        let state: Cluster = match saved {
            Some((1, body)) => serde_json::from_str(&body).map_err(|e| e.to_string())?,
            Some(_) => return Err("unsupported database schema version".into()),
            None => initial,
        };
        if state.mode != expected_mode {
            return Err(
                "database belongs to a different execution mode; use a separate database".into(),
            );
        }
        let mut store = Self { connection, state };
        store.update(|_| Ok(()))?;
        Ok(store)
    }

    pub fn state(&self) -> &Cluster {
        &self.state
    }

    pub fn update<T>(
        &mut self,
        change: impl FnOnce(&mut Cluster) -> Result<T, String>,
    ) -> Result<T, String> {
        let mut candidate = self.state.clone();
        let result = change(&mut candidate)?;
        candidate.validate()?;
        let body = serde_json::to_string(&candidate).map_err(|e| e.to_string())?;
        self.connection
            .execute(
                "INSERT INTO snapshot VALUES (1, 1, ?1)
            ON CONFLICT(id) DO UPDATE SET body=excluded.body",
                [body],
            )
            .map_err(|e| e.to_string())?;
        self.state = candidate;
        Ok(result)
    }
}
