"""
Supply chain data schemas for RiskX platform.
Pydantic models for supply chain entities, logistics, and operational data validation.
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, date
from decimal import Decimal
from enum import Enum

from ...utils.constants import BusinessRules


class SupplyChainNodeType(str, Enum):
    """Types of supply chain nodes."""
    SUPPLIER = "supplier"
    MANUFACTURER = "manufacturer"
    DISTRIBUTOR = "distributor"
    RETAILER = "retailer"
    LOGISTICS_PROVIDER = "logistics_provider"
    WAREHOUSE = "warehouse"
    PORT = "port"
    CUSTOMS = "customs"
    END_CUSTOMER = "end_customer"


class TransportationMode(str, Enum):
    """Modes of transportation."""
    ROAD = "road"
    RAIL = "rail"
    AIR = "air"
    SEA = "sea"
    PIPELINE = "pipeline"
    INTERMODAL = "intermodal"


class ShipmentStatus(str, Enum):
    """Status of shipments."""
    PLANNED = "planned"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    DELAYED = "delayed"
    CANCELLED = "cancelled"
    LOST = "lost"
    DAMAGED = "damaged"


class RiskLevel(str, Enum):
    """Supply chain risk levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class InventoryStatus(str, Enum):
    """Inventory status classifications."""
    IN_STOCK = "in_stock"
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"
    BACKORDERED = "backordered"
    DISCONTINUED = "discontinued"


class SupplyChainNode(BaseModel):
    """Base model for supply chain entities."""
    
    node_id: str = Field(..., description="Unique identifier for the node")
    name: str = Field(..., description="Name of the supply chain entity")
    node_type: SupplyChainNodeType = Field(..., description="Type of supply chain node")
    address: Optional[str] = Field(None, description="Physical address")
    city: Optional[str] = Field(None, description="City location")
    state_province: Optional[str] = Field(None, description="State or province")
    country: str = Field(..., description="Country location")
    postal_code: Optional[str] = Field(None, description="Postal/ZIP code")
    latitude: Optional[Decimal] = Field(None, description="Geographic latitude")
    longitude: Optional[Decimal] = Field(None, description="Geographic longitude")
    contact_info: Optional[Dict[str, str]] = Field(default_factory=dict, description="Contact information")
    certification: Optional[List[str]] = Field(default_factory=list, description="Industry certifications")
    capacity: Optional[Decimal] = Field(None, description="Production or handling capacity")
    established_date: Optional[date] = Field(None, description="Date entity was established")
    
    @validator('node_id')
    def validate_node_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Node ID cannot be empty')
        return v.strip().upper()
    
    @validator('country')
    def validate_country(cls, v):
        if len(v) < 2:
            raise ValueError('Country must be at least 2 characters')
        return v.upper()
    
    @validator('latitude')
    def validate_latitude(cls, v):
        if v is not None and (v < -90 or v > 90):
            raise ValueError('Latitude must be between -90 and 90')
        return v
    
    @validator('longitude')
    def validate_longitude(cls, v):
        if v is not None and (v < -180 or v > 180):
            raise ValueError('Longitude must be between -180 and 180')
        return v


class Supplier(BaseModel):
    """Supplier-specific data and metrics."""
    
    node: SupplyChainNode = Field(..., description="Basic node information")
    supplier_code: str = Field(..., description="Internal supplier code")
    tier: int = Field(..., description="Supplier tier (1, 2, 3, etc.)")
    products_supplied: List[str] = Field(..., description="List of products/services supplied")
    annual_revenue: Optional[Decimal] = Field(None, description="Annual revenue")
    employee_count: Optional[int] = Field(None, description="Number of employees")
    quality_rating: Optional[Decimal] = Field(None, description="Quality rating score")
    delivery_performance: Optional[Decimal] = Field(None, description="On-time delivery percentage")
    payment_terms: Optional[str] = Field(None, description="Payment terms")
    contract_expiry: Optional[date] = Field(None, description="Contract expiration date")
    risk_score: Optional[Decimal] = Field(None, description="Supplier risk score")
    backup_suppliers: Optional[List[str]] = Field(default_factory=list, description="Alternative suppliers")
    
    @validator('tier')
    def validate_tier(cls, v):
        if v < 1 or v > 10:
            raise ValueError('Supplier tier must be between 1 and 10')
        return v
    
    @validator('quality_rating', 'delivery_performance')
    def validate_percentages(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError('Ratings and percentages must be between 0 and 100')
        return v


class Manufacturer(BaseModel):
    """Manufacturer-specific data and metrics."""
    
    node: SupplyChainNode = Field(..., description="Basic node information")
    facility_code: str = Field(..., description="Manufacturing facility code")
    production_capacity: Decimal = Field(..., description="Maximum production capacity")
    current_utilization: Decimal = Field(..., description="Current capacity utilization")
    products_manufactured: List[str] = Field(..., description="List of manufactured products")
    equipment_age: Optional[Decimal] = Field(None, description="Average equipment age in years")
    automation_level: Optional[Decimal] = Field(None, description="Level of automation (0-100)")
    quality_certifications: Optional[List[str]] = Field(default_factory=list, description="Quality certifications")
    environmental_compliance: Optional[bool] = Field(None, description="Environmental compliance status")
    safety_incidents: Optional[int] = Field(None, description="Number of safety incidents")
    downtime_hours: Optional[Decimal] = Field(None, description="Unplanned downtime hours")
    efficiency_score: Optional[Decimal] = Field(None, description="Overall efficiency score")
    
    @validator('current_utilization')
    def validate_utilization(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Capacity utilization must be between 0 and 100')
        return v
    
    @validator('automation_level', 'efficiency_score')
    def validate_scores(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError('Scores must be between 0 and 100')
        return v


class Distributor(BaseModel):
    """Distributor-specific data and metrics."""
    
    node: SupplyChainNode = Field(..., description="Basic node information")
    distributor_code: str = Field(..., description="Distributor identification code")
    coverage_area: List[str] = Field(..., description="Geographic coverage areas")
    storage_capacity: Decimal = Field(..., description="Total storage capacity")
    current_inventory: Decimal = Field(..., description="Current inventory level")
    throughput_capacity: Decimal = Field(..., description="Daily throughput capacity")
    distribution_channels: List[str] = Field(..., description="Distribution channels used")
    delivery_fleet_size: Optional[int] = Field(None, description="Number of delivery vehicles")
    warehouse_automation: Optional[Decimal] = Field(None, description="Warehouse automation level")
    order_accuracy: Optional[Decimal] = Field(None, description="Order accuracy percentage")
    average_delivery_time: Optional[Decimal] = Field(None, description="Average delivery time in hours")
    cost_per_shipment: Optional[Decimal] = Field(None, description="Average cost per shipment")
    
    @validator('current_inventory')
    def validate_inventory(cls, v):
        if v < 0:
            raise ValueError('Current inventory cannot be negative')
        return v
    
    @validator('order_accuracy')
    def validate_accuracy(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError('Order accuracy must be between 0 and 100')
        return v


class Retailer(BaseModel):
    """Retailer-specific data and metrics."""
    
    node: SupplyChainNode = Field(..., description="Basic node information")
    store_code: str = Field(..., description="Store identification code")
    store_format: str = Field(..., description="Store format type")
    sales_area: Optional[Decimal] = Field(None, description="Sales floor area in square feet")
    customer_traffic: Optional[int] = Field(None, description="Daily customer traffic")
    sales_volume: Optional[Decimal] = Field(None, description="Daily sales volume")
    inventory_turnover: Optional[Decimal] = Field(None, description="Inventory turnover ratio")
    stockout_rate: Optional[Decimal] = Field(None, description="Stockout rate percentage")
    customer_satisfaction: Optional[Decimal] = Field(None, description="Customer satisfaction score")
    employee_count: Optional[int] = Field(None, description="Number of employees")
    operating_hours: Optional[str] = Field(None, description="Daily operating hours")
    seasonal_factor: Optional[Decimal] = Field(None, description="Seasonal sales variation factor")
    
    @validator('stockout_rate', 'customer_satisfaction')
    def validate_rates(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError('Rates and satisfaction scores must be between 0 and 100')
        return v


class TransportationData(BaseModel):
    """Transportation and logistics data."""
    
    shipment_id: str = Field(..., description="Unique shipment identifier")
    transportation_mode: TransportationMode = Field(..., description="Mode of transportation")
    carrier: str = Field(..., description="Transportation carrier")
    origin: str = Field(..., description="Origin location")
    destination: str = Field(..., description="Destination location")
    departure_time: Optional[datetime] = Field(None, description="Scheduled departure time")
    arrival_time: Optional[datetime] = Field(None, description="Scheduled arrival time")
    actual_departure: Optional[datetime] = Field(None, description="Actual departure time")
    actual_arrival: Optional[datetime] = Field(None, description="Actual arrival time")
    distance: Optional[Decimal] = Field(None, description="Distance in miles/kilometers")
    fuel_consumption: Optional[Decimal] = Field(None, description="Fuel consumption")
    cost: Optional[Decimal] = Field(None, description="Transportation cost")
    carbon_emissions: Optional[Decimal] = Field(None, description="Carbon emissions")
    
    @validator('distance', 'fuel_consumption', 'cost', 'carbon_emissions')
    def validate_positive_values(cls, v):
        if v is not None and v < 0:
            raise ValueError('Values cannot be negative')
        return v


class InventoryData(BaseModel):
    """Inventory level and management data."""
    
    item_id: str = Field(..., description="Item identifier")
    location_id: str = Field(..., description="Storage location identifier")
    product_name: str = Field(..., description="Product name")
    category: str = Field(..., description="Product category")
    current_stock: int = Field(..., description="Current stock level")
    reserved_stock: Optional[int] = Field(None, description="Reserved stock level")
    available_stock: int = Field(..., description="Available stock level")
    safety_stock: Optional[int] = Field(None, description="Safety stock level")
    reorder_point: Optional[int] = Field(None, description="Reorder point")
    maximum_stock: Optional[int] = Field(None, description="Maximum stock level")
    unit_cost: Optional[Decimal] = Field(None, description="Unit cost")
    total_value: Optional[Decimal] = Field(None, description="Total inventory value")
    last_movement_date: Optional[date] = Field(None, description="Last inventory movement date")
    expiry_date: Optional[date] = Field(None, description="Product expiry date")
    status: InventoryStatus = Field(default=InventoryStatus.IN_STOCK, description="Inventory status")
    
    @validator('current_stock', 'available_stock')
    def validate_stock_levels(cls, v):
        if v < 0:
            raise ValueError('Stock levels cannot be negative')
        return v
    
    @root_validator
    def validate_stock_relationships(cls, values):
        current = values.get('current_stock', 0)
        reserved = values.get('reserved_stock', 0)
        available = values.get('available_stock', 0)
        
        if reserved and current >= 0 and available >= 0:
            expected_available = current - reserved
            if available != expected_available:
                values['available_stock'] = max(0, expected_available)
        
        return values


class ShipmentData(BaseModel):
    """Shipment tracking and status data."""
    
    shipment_id: str = Field(..., description="Unique shipment identifier")
    order_id: str = Field(..., description="Related order identifier")
    origin_node: str = Field(..., description="Origin node identifier")
    destination_node: str = Field(..., description="Destination node identifier")
    products: List[Dict[str, Any]] = Field(..., description="List of products in shipment")
    status: ShipmentStatus = Field(..., description="Current shipment status")
    total_weight: Optional[Decimal] = Field(None, description="Total weight")
    total_volume: Optional[Decimal] = Field(None, description="Total volume")
    total_value: Optional[Decimal] = Field(None, description="Total value of goods")
    packaging_type: Optional[str] = Field(None, description="Type of packaging")
    special_handling: Optional[List[str]] = Field(default_factory=list, description="Special handling requirements")
    insurance_value: Optional[Decimal] = Field(None, description="Insurance value")
    tracking_number: Optional[str] = Field(None, description="Carrier tracking number")
    estimated_delivery: Optional[datetime] = Field(None, description="Estimated delivery time")
    actual_delivery: Optional[datetime] = Field(None, description="Actual delivery time")
    delays: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Delay incidents")
    
    @validator('total_weight', 'total_volume', 'total_value', 'insurance_value')
    def validate_positive_values(cls, v):
        if v is not None and v < 0:
            raise ValueError('Values cannot be negative')
        return v


class SupplyChainRisk(BaseModel):
    """Supply chain risk assessment data."""
    
    risk_id: str = Field(..., description="Unique risk identifier")
    node_id: str = Field(..., description="Associated node identifier")
    risk_type: str = Field(..., description="Type of risk")
    risk_category: str = Field(..., description="Risk category")
    risk_level: RiskLevel = Field(..., description="Risk severity level")
    probability: Decimal = Field(..., description="Probability of occurrence")
    impact_score: Decimal = Field(..., description="Impact severity score")
    risk_score: Decimal = Field(..., description="Overall risk score")
    description: str = Field(..., description="Risk description")
    potential_impact: Optional[str] = Field(None, description="Potential business impact")
    mitigation_measures: Optional[List[str]] = Field(default_factory=list, description="Risk mitigation measures")
    owner: Optional[str] = Field(None, description="Risk owner")
    detection_date: date = Field(..., description="Risk detection date")
    review_date: Optional[date] = Field(None, description="Next review date")
    status: str = Field(default="active", description="Risk status")
    
    @validator('probability', 'impact_score', 'risk_score')
    def validate_scores(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Scores must be between 0 and 100')
        return v


class LogisticsData(BaseModel):
    """Logistics performance and metrics data."""
    
    facility_id: str = Field(..., description="Logistics facility identifier")
    date: date = Field(..., description="Data observation date")
    throughput: Optional[int] = Field(None, description="Daily throughput volume")
    utilization_rate: Optional[Decimal] = Field(None, description="Facility utilization rate")
    processing_time: Optional[Decimal] = Field(None, description="Average processing time")
    accuracy_rate: Optional[Decimal] = Field(None, description="Order accuracy rate")
    damage_rate: Optional[Decimal] = Field(None, description="Damage rate percentage")
    cost_per_unit: Optional[Decimal] = Field(None, description="Cost per unit processed")
    employee_productivity: Optional[Decimal] = Field(None, description="Employee productivity score")
    equipment_uptime: Optional[Decimal] = Field(None, description="Equipment uptime percentage")
    safety_incidents: Optional[int] = Field(None, description="Number of safety incidents")
    energy_consumption: Optional[Decimal] = Field(None, description="Energy consumption")
    
    @validator('utilization_rate', 'accuracy_rate', 'damage_rate', 'equipment_uptime')
    def validate_rates(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError('Rates must be between 0 and 100')
        return v


class SupplyChainMetrics(BaseModel):
    """Supply chain performance metrics."""
    
    metric_date: date = Field(..., description="Metrics calculation date")
    overall_performance: Decimal = Field(..., description="Overall supply chain performance score")
    cost_efficiency: Optional[Decimal] = Field(None, description="Cost efficiency score")
    delivery_performance: Optional[Decimal] = Field(None, description="Delivery performance score")
    quality_performance: Optional[Decimal] = Field(None, description="Quality performance score")
    flexibility_score: Optional[Decimal] = Field(None, description="Supply chain flexibility score")
    resilience_score: Optional[Decimal] = Field(None, description="Supply chain resilience score")
    sustainability_score: Optional[Decimal] = Field(None, description="Sustainability score")
    cash_to_cash_cycle: Optional[Decimal] = Field(None, description="Cash-to-cash cycle time")
    perfect_order_rate: Optional[Decimal] = Field(None, description="Perfect order fulfillment rate")
    supply_chain_costs: Optional[Decimal] = Field(None, description="Total supply chain costs")
    inventory_turns: Optional[Decimal] = Field(None, description="Inventory turnover ratio")
    
    @validator('overall_performance', 'cost_efficiency', 'delivery_performance', 
              'quality_performance', 'flexibility_score', 'resilience_score', 
              'sustainability_score', 'perfect_order_rate')
    def validate_performance_scores(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError('Performance scores must be between 0 and 100')
        return v


class SupplyChainSummary(BaseModel):
    """Summary statistics for supply chain data."""
    
    summary_period_start: date = Field(..., description="Summary period start date")
    summary_period_end: date = Field(..., description="Summary period end date")
    total_nodes: int = Field(..., description="Total number of supply chain nodes")
    total_shipments: int = Field(..., description="Total number of shipments")
    average_delivery_time: Optional[Decimal] = Field(None, description="Average delivery time")
    on_time_delivery_rate: Optional[Decimal] = Field(None, description="On-time delivery rate")
    total_costs: Optional[Decimal] = Field(None, description="Total supply chain costs")
    average_risk_score: Optional[Decimal] = Field(None, description="Average risk score across nodes")
    high_risk_nodes: int = Field(default=0, description="Number of high-risk nodes")
    disruption_incidents: int = Field(default=0, description="Number of disruption incidents")
    cost_savings: Optional[Decimal] = Field(None, description="Cost savings achieved")
    efficiency_improvement: Optional[Decimal] = Field(None, description="Efficiency improvement percentage")
    
    @validator('total_nodes', 'total_shipments', 'high_risk_nodes', 'disruption_incidents')
    def validate_counts(cls, v):
        if v < 0:
            raise ValueError('Counts cannot be negative')
        return v