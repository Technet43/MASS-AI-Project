package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strconv"
	"sync"
	"syscall"
	"time"

	mqtt "github.com/eclipse/paho.mqtt.golang"
	_ "github.com/lib/pq"
)

const (
	defaultBrokerHost = "127.0.0.1"
	defaultBrokerPort = 1883
	defaultTopic      = "mass_ai/telemetry"

	// DB pool
	dbMaxOpenConns    = 10
	dbMaxIdleConns    = 5
	dbConnMaxLifetime = 5 * time.Minute

	// Batch insert: flush when buffer reaches batchSize OR batchFlushInterval.
	batchSize          = 50
	batchFlushInterval = 500 * time.Millisecond
)

// TelemetryMessage represents one incoming smart meter event.
type TelemetryMessage struct {
	MeterID     string  `json:"meter_id"`
	Timestamp   string  `json:"timestamp"`
	Voltage     float64 `json:"voltage"`
	Current     float64 `json:"current"`
	ActivePower float64 `json:"active_power"`
}

// batchBuffer accumulates rows before a bulk INSERT.
type batchBuffer struct {
	mu   sync.Mutex
	rows []TelemetryMessage
}

func (b *batchBuffer) add(msg TelemetryMessage) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.rows = append(b.rows, msg)
	return len(b.rows) >= batchSize
}

func (b *batchBuffer) drain() []TelemetryMessage {
	b.mu.Lock()
	defer b.mu.Unlock()
	if len(b.rows) == 0 {
		return nil
	}
	out := b.rows
	b.rows = make([]TelemetryMessage, 0, batchSize)
	return out
}

func main() {
	logger := log.New(os.Stdout, "", 0)

	// ── Postgres ──────────────────────────────────────────────────────────────
	db, err := openDB(logger)
	if err != nil {
		logger.Fatalf("[%s] db connect failed: %v", nowMillis(), err)
	}
	defer db.Close()

	buf := &batchBuffer{rows: make([]TelemetryMessage, 0, batchSize)}

	// Background goroutine: flush buffer on a timer.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go flushLoop(ctx, logger, db, buf)

	// ── MQTT ──────────────────────────────────────────────────────────────────
	broker := brokerURL()
	topic := getenv("MQTT_TOPIC", defaultTopic)
	clientID := fmt.Sprintf("mass-ai-gateway-%d", time.Now().UnixNano())

	opts := mqtt.NewClientOptions().
		AddBroker(broker).
		SetClientID(clientID).
		SetAutoReconnect(true).
		SetConnectRetry(true).
		SetConnectRetryInterval(2 * time.Second).
		SetKeepAlive(30 * time.Second).
		SetPingTimeout(10 * time.Second).
		SetOrderMatters(false).
		SetResumeSubs(true)

	opts.SetConnectionLostHandler(func(_ mqtt.Client, err error) {
		logger.Printf("[%s] mqtt connection lost: %v", nowMillis(), err)
	})

	opts.SetOnConnectHandler(func(client mqtt.Client) {
		logger.Printf("[%s] connected broker=%s topic=%s", nowMillis(), broker, topic)
		token := client.Subscribe(topic, 1, func(_ mqtt.Client, msg mqtt.Message) {
			handleMessage(logger, db, buf, msg)
		})
		if ok := token.WaitTimeout(5 * time.Second); !ok {
			logger.Printf("[%s] subscribe timeout topic=%s", nowMillis(), topic)
			return
		}
		if err := token.Error(); err != nil {
			logger.Printf("[%s] subscribe error topic=%s err=%v", nowMillis(), topic, err)
			return
		}
		logger.Printf("[%s] subscription active topic=%s", nowMillis(), topic)
	})

	client := mqtt.NewClient(opts)
	connectToken := client.Connect()
	if ok := connectToken.WaitTimeout(10 * time.Second); !ok {
		logger.Fatalf("[%s] mqtt connect timeout broker=%s", nowMillis(), broker)
	}
	if err := connectToken.Error(); err != nil {
		logger.Fatalf("[%s] mqtt connect failed broker=%s err=%v", nowMillis(), broker, err)
	}

	<-ctx.Done()
	logger.Printf("[%s] shutdown — flushing remaining buffer…", nowMillis())
	client.Disconnect(250)
	flushBatch(logger, db, buf.drain())
}

// flushLoop flushes the buffer every batchFlushInterval regardless of size.
func flushLoop(ctx context.Context, logger *log.Logger, db *sql.DB, buf *batchBuffer) {
	ticker := time.NewTicker(batchFlushInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if rows := buf.drain(); len(rows) > 0 {
				flushBatch(logger, db, rows)
			}
		}
	}
}

// handleMessage validates and buffers an incoming MQTT message.
// If the buffer is full it triggers an immediate synchronous flush.
func handleMessage(logger *log.Logger, db *sql.DB, buf *batchBuffer, msg mqtt.Message) {
	var payload TelemetryMessage
	if err := json.Unmarshal(msg.Payload(), &payload); err != nil {
		logger.Printf("[%s] parse error err=%v raw=%s", nowMillis(), err, string(msg.Payload()))
		return
	}
	if err := validateTelemetry(payload); err != nil {
		logger.Printf("[%s] validation error meter=%s err=%v", nowMillis(), payload.MeterID, err)
		return
	}
	if _, err := time.Parse(time.RFC3339Nano, payload.Timestamp); err != nil {
		logger.Printf("[%s] invalid timestamp meter=%s err=%v", nowMillis(), payload.MeterID, err)
		return
	}

	if full := buf.add(payload); full {
		rows := buf.drain()
		go flushBatch(logger, db, rows) // non-blocking: let the MQTT handler return fast
	}
}

// flushBatch performs a single multi-value INSERT for a slice of messages.
func flushBatch(logger *log.Logger, db *sql.DB, rows []TelemetryMessage) {
	if len(rows) == 0 {
		return
	}

	// Build: INSERT … VALUES ($1,$2,$3,$4), ($5,$6,$7,$8), …
	query := "INSERT INTO raw_telemetry (meter_id, voltage, current, active_power, received_at) VALUES "
	args := make([]any, 0, len(rows)*4)
	for i, r := range rows {
		if i > 0 {
			query += ","
		}
		base := i * 4
		query += fmt.Sprintf("($%d,$%d,$%d,$%d,NOW())", base+1, base+2, base+3, base+4)
		args = append(args, r.MeterID, r.Voltage, r.Current, r.ActivePower)
	}

	if _, err := db.Exec(query, args...); err != nil {
		logger.Printf("[%s] batch insert error rows=%d err=%v", nowMillis(), len(rows), err)
		return
	}
	logger.Printf("[%s] batch inserted rows=%d", nowMillis(), len(rows))
}

// openDB opens a Postgres connection pool and waits until ready.
func openDB(logger *log.Logger) (*sql.DB, error) {
	dsn := fmt.Sprintf(
		"host=%s port=%s dbname=%s user=%s password=%s sslmode=disable",
		getenv("DB_HOST", "127.0.0.1"),
		getenv("DB_PORT", "5432"),
		getenv("DB_NAME", "mass_ai"),
		getenv("DB_USER", "mass_ai"),
		getenv("DB_PASSWORD", "mass_ai_secret"),
	)

	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return nil, fmt.Errorf("sql.Open: %w", err)
	}
	db.SetMaxOpenConns(dbMaxOpenConns)
	db.SetMaxIdleConns(dbMaxIdleConns)
	db.SetConnMaxLifetime(dbConnMaxLifetime)

	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		if err = db.Ping(); err == nil {
			logger.Printf("[%s] postgres ready host=%s db=%s", nowMillis(),
				getenv("DB_HOST", "127.0.0.1"), getenv("DB_NAME", "mass_ai"))
			return db, nil
		}
		logger.Printf("[%s] waiting for postgres: %v", nowMillis(), err)
		time.Sleep(2 * time.Second)
	}
	return nil, fmt.Errorf("postgres not ready after 30s: %w", err)
}

func validateTelemetry(message TelemetryMessage) error {
	if message.MeterID == "" {
		return errors.New("meter_id is empty")
	}
	if message.Voltage <= 0 {
		return errors.New("voltage must be positive")
	}
	if message.Current < 0 {
		return errors.New("current must be zero or positive")
	}
	if message.ActivePower < 0 {
		return errors.New("active_power must be zero or positive")
	}
	return nil
}

func brokerURL() string {
	host := getenv("MQTT_HOST", defaultBrokerHost)
	portValue := getenv("MQTT_PORT", strconv.Itoa(defaultBrokerPort))
	port, err := strconv.Atoi(portValue)
	if err != nil {
		log.Fatalf("[%s] invalid MQTT_PORT=%q: %v", nowMillis(), portValue, err)
	}
	return fmt.Sprintf("tcp://%s:%d", host, port)
}

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func nowMillis() string {
	return time.Now().UTC().Format("2006-01-02T15:04:05.000Z07:00")
}
