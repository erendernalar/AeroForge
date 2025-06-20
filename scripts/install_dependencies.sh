#!/bin/bash

echo "📦 Installing AeroForge dependencies..."

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ pip not found. Please install Python and pip first."
    exit 1
fi

# Install Python packages
echo "📚 Installing Python packages..."
pip install -r requirements.txt

echo "🔍 Checking XFOIL..."
python scripts/validate_xfoil.py

if [ $? -eq 0 ]; then
    echo "✅ XFOIL is working correctly"
else
    echo "⚠️  XFOIL needs to be installed manually"
    echo "   Download from: https://web.mit.edu/drela/Public/web/xfoil/"
fi

echo "🧪 Running basic tests..."
python -m pytest tests/test_basic.py -v

echo "🎉 AeroForge setup complete!"
echo ""
echo "Next steps:"
echo "  1. Install XFOIL if not already installed"
echo "  2. Run demo: python main.py --demo"
echo "  3. Start optimizing: python main.py --config config/enhanced_config.json"
