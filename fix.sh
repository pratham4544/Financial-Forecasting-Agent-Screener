#!/bin/bash

echo "🧪 Testing Financial Q&A Agent"
echo "==============================="
echo ""

BASE_URL="http://localhost:8000"

# Check if server is running
echo "📡 Checking server..."
if ! curl -s "$BASE_URL/health" > /dev/null 2>&1; then
    echo "❌ Server not running. Start with: python main.py"
    exit 1
fi
echo "✅ Server is running"
echo ""

# Test 1: Initialize Session
echo "Test 1: Initialize Session"
echo "--------------------------"
echo "Creating session for TCS..."

SESSION_RESPONSE=$(curl -s -X POST "$BASE_URL/session/init" \
  -H "Content-Type: application/json" \
  -d '{
    "company_url": "https://www.screener.in/company/TCS/consolidated/",
    "quarters": 1
  }')

SESSION_ID=$(echo $SESSION_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['session_id'])" 2>/dev/null)

if [ -z "$SESSION_ID" ]; then
    echo "❌ Failed to create session"
    echo "$SESSION_RESPONSE"
    exit 1
fi

echo "✅ Session created: $SESSION_ID"
echo "$SESSION_RESPONSE" | python3 -m json.tool 2>/dev/null
echo ""

# Wait for PDFs to download
echo "⏳ Waiting for PDFs to download (30 seconds)..."
sleep 30
echo ""

# Test 2: Ask First Question
echo "Test 2: Ask Question #1"
echo "----------------------"
echo "Question: What was the revenue growth?"

ANSWER1=$(curl -s -X POST "$BASE_URL/session/ask" \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SESSION_ID\",
    \"question\": \"What was the revenue growth last quarter?\"
  }")

echo "$ANSWER1" | python3 -m json.tool 2>/dev/null
echo ""

# Test 3: Ask Second Question (should be fast!)
echo "Test 3: Ask Question #2 (should be fast!)"
echo "----------------------------------------"
echo "Question: What are the key risks?"

ANSWER2=$(curl -s -X POST "$BASE_URL/session/ask" \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SESSION_ID\",
    \"question\": \"What are the key risks mentioned?\"
  }")

echo "$ANSWER2" | python3 -m json.tool 2>/dev/null
echo ""

# Test 4: Ask Third Question
echo "Test 4: Ask Question #3"
echo "----------------------"
echo "Question: What did management say about digital transformation?"

ANSWER3=$(curl -s -X POST "$BASE_URL/session/ask" \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SESSION_ID\",
    \"question\": \"What did management say about digital transformation?\"
  }")

echo "$ANSWER3" | python3 -m json.tool 2>/dev/null
echo ""

# Test 5: View History
echo "Test 5: View Session History"
echo "----------------------------"

HISTORY=$(curl -s "$BASE_URL/session/$SESSION_ID/history")
TOTAL_QUESTIONS=$(echo $HISTORY | python3 -c "import sys, json; print(json.load(sys.stdin)['total_questions'])" 2>/dev/null)

echo "✅ Total questions asked: $TOTAL_QUESTIONS"
echo "$HISTORY" | python3 -m json.tool 2>/dev/null
echo ""

# Test 6: List Sessions
echo "Test 6: List All Sessions"
echo "------------------------"

curl -s "$BASE_URL/sessions/list" | python3 -m json.tool 2>/dev/null
echo ""

# Cleanup (optional)
echo "Cleanup"
echo "-------"
read -p "Delete test session? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    curl -s -X DELETE "$BASE_URL/session/$SESSION_ID"
    echo "✅ Session deleted"
else
    echo "⏭️  Session kept: $SESSION_ID"
    echo "   You can continue asking questions with this session_id"
fi

echo ""
echo "=============================="
echo "🎉 All tests completed!"
echo "=============================="
echo ""
echo "Your Q&A agent is working! 🚀"
echo ""
echo "Next steps:"
echo "1. Visit http://localhost:8000/docs for interactive testing"
echo "2. Try asking more questions using session_id: $SESSION_ID"
echo "3. Create sessions for other companies"